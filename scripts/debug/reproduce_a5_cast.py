"""Manual single-device reproducer for the A5 eager packing failure; no training."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch._decomp.decompositions import _chunk_cat, _pad_chunk


# Ordered shapes from the failed FSDP call (545), including both final norms.
SHAPES = [
    (8192, 2048),
    (512, 2048),
    (512, 2048),
    (2048, 4096),
    (256,),
    (256,),
    (256, 2048),
    (256, 1024, 2048),
    (256, 2048, 512),
    (512, 2048),
    (512, 2048),
    (2048, 512),
    (1, 2048),
    (2048,),
    (2048,),
]


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="cpu or npu:N; one device, no distributed initialization")
    parser.add_argument("--mode", choices=["eager", "cast", "precast-cat", "mixed-cat"], required=True)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--allocation", choices=["reuse", "fresh"], default="reuse")
    parser.add_argument("--size", choices=["real", "tiny"], default="real")
    parser.add_argument("--tensor-index", type=int, choices=range(len(SHAPES)), help="cast mode only, zero based")
    parser.add_argument("--log-dir", type=Path, required=True, help="must not already exist")
    args = parser.parse_args()
    if args.device != "cpu" and not (args.device.startswith("npu:") and args.device[4:].isdigit()):
        parser.error("device must be cpu or npu:N")
    if args.tensor_index is not None and args.mode != "cast":
        parser.error("--tensor-index is only supported in cast mode")
    if torch.__version__.split("+")[0].split(".")[:2] != ["2", "10"]:
        parser.error(f"this private decomposition reproducer requires torch 2.10, found {torch.__version__}")

    args.log_dir.mkdir(parents=True, exist_ok=False)
    with (args.log_dir / "events.jsonl").open("x", buffering=1) as log:
        iteration = 0
        active_phase = "setup"

        def emit(event, **data):
            record = {"event": event, "iteration": iteration, "phase": active_phase, "time": time.time(), **data}
            line = json.dumps(record)
            log.write(line + "\n")
            print(line, flush=True)

        def synchronize():
            if args.device.startswith("npu:"):
                torch.npu.synchronize()

        def phase(name, function, *args, **kwargs):
            nonlocal active_phase
            active_phase = name
            emit("begin")
            start = time.monotonic()
            value = function(*args, **kwargs)
            synchronize()
            emit("complete", seconds=time.monotonic() - start)
            return value

        def metadata(tensor):
            return {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "data_ptr": tensor.data_ptr(),
                "storage_offset": tensor.storage_offset(),
                "bytes": tensor.numel() * tensor.element_size(),
            }

        try:
            emit(
                "config",
                options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                torch=torch.__version__,
                environment={
                    k: os.environ.get(k)
                    for k in ("ASCEND_LAUNCH_BLOCKING", "MULTI_STREAM_MEMORY_REUSE", "ASCEND_RT_VISIBLE_DEVICES")
                },
            )
            if args.device.startswith("npu:"):
                import torch_npu

                emit("runtime", torch_npu=torch_npu.__version__)
                phase("device_init", torch.npu.set_device, args.device)

            indices = list(range(len(SHAPES))) if args.tensor_index is None else [args.tensor_index]
            shapes = [SHAPES[i] if args.size == "real" else tuple(min(d, 4) for d in SHAPES[i]) for i in indices]
            chunks = 8
            widths = [math.ceil(shape[0] / chunks) * math.prod(shape[1:]) for shape in shapes]
            width = sum(widths)

            def allocate():
                inputs = []
                for index, shape in zip(indices, shapes):
                    # Match the observed (1, 2048) tensor's singleton stride too.
                    tensor = (
                        torch.empty_strided(shape, (1, 1), device=args.device, dtype=torch.bfloat16)
                        if index == 12
                        else torch.empty(shape, device=args.device, dtype=torch.bfloat16)
                    )
                    tensor.fill_((index + 1) / 16)
                    inputs.append(tensor)
                out = (
                    None
                    if args.mode == "cast"
                    else torch.empty((chunks, width), device=args.device, dtype=torch.float32)
                )
                return inputs, out

            def verify_packed(tensor, original_shape, index):
                # Small deterministic samples only; no multi-GiB host copy.
                rows_per_chunk = math.ceil(original_shape[0] / chunks)
                row_width = math.prod(original_shape[1:])
                samples, expected = [], []
                for rank in range(chunks):
                    valid = max(0, min(rows_per_chunk, original_shape[0] - rank * rows_per_chunk)) * row_width
                    for col in sorted({0, max(0, valid - 1), tensor.shape[1] - 1}):
                        samples.append(tensor[rank, col])
                        expected.append((index + 1) / 16 if col < valid else 0.0)
                values = torch.stack(samples).cpu()
                torch.testing.assert_close(values, torch.tensor(expected, dtype=torch.float32), rtol=0, atol=0)

            def pad_inputs(inputs):
                return [_pad_chunk(t, 0, chunks) for t in inputs]

            def verify_output(out):
                offset = 0
                for index, shape, part_width in zip(indices, shapes, widths):
                    verify_packed(out[:, offset : offset + part_width], shape, index)
                    offset += part_width

            with torch.no_grad():
                inputs, out = None, None
                for attempt in range(1, args.iterations + 1):
                    iteration = attempt
                    if inputs is None or args.allocation == "fresh":
                        # Release references first; fresh allocation may reuse allocator addresses.
                        inputs, out = None, None
                        inputs, out = phase("allocate_and_fill", allocate)
                    emit(
                        "buffers", inputs=[metadata(t) for t in inputs], out=metadata(out) if out is not None else None
                    )
                    if out is not None:
                        phase("poison_output", out.fill_, float("nan"))
                    if args.mode == "eager":
                        phase("eager_pack", _chunk_cat, inputs, dim=0, num_chunks=chunks, out=out)
                    else:
                        packed = phase("pad_and_reshape", pad_inputs, inputs)
                        emit("packed_buffers", tensors=[metadata(t) for t in packed])
                        if args.mode == "mixed-cat":
                            phase("mixed_cat", torch.cat, packed, dim=1, out=out)
                        elif args.mode == "cast":
                            for index, shape, tensor in zip(indices, shapes, packed):
                                converted = phase(f"cast_tensor_{index}", tensor.to, torch.float32)
                                emit("cast_output", tensor_index=index, tensor=metadata(converted))
                                phase(f"check_tensor_{index}", verify_packed, converted, shape, index)
                                del converted
                            del tensor
                        else:
                            converted = []
                            for index, tensor in zip(indices, packed):
                                converted.append(phase(f"cast_tensor_{index}", tensor.to, torch.float32))
                            phase("fp32_cat", torch.cat, converted, dim=1, out=out)
                            del tensor, converted
                        del packed
                    if out is not None:
                        phase("check_output", verify_output, out)
                emit("pass", completed_iterations=args.iterations)
        except BaseException as error:
            # Do not issue further device calls after a kernel failure.
            emit("failed", error_type=type(error).__name__, message=str(error))
            raise


if __name__ == "__main__":
    main()
