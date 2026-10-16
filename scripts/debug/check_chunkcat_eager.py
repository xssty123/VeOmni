"""Manual packing parity check. Defaults to CPU; NPU requires --device npu:0."""

import argparse
import importlib.util
import math
from pathlib import Path

import torch


def load_entry():
    path = Path(__file__).resolve().parents[2] / "tasks/train_vlm_chunkcat_eager.py"
    spec = importlib.util.spec_from_file_location("chunkcat_eager_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    entry = load_entry()
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives

    original = entry.install_chunkcat_eager()
    original_chunk_cat = torch._chunk_cat
    cases = [
        [(1,), (3,)],
        [(5,), (11,)],
        [(8,), (16,)],
        [(5, 3), (2, 7)],
        [(3, 2, 5), (7, 4, 2)],
        [(5,), (3, 2), (2, 3, 4)],
    ]
    checked = 0
    try:
        for shapes in cases:
            for noncontiguous in (False, True):
                for input_dtype, output_dtype in (
                    (torch.bfloat16, torch.float32),
                    (torch.bfloat16, torch.bfloat16),
                    (torch.float32, torch.float32),
                ):
                    inputs = []
                    for shape in shapes:
                        # Slicing makes even one-dimensional input noncontiguous.
                        x = (torch.arange(math.prod(shape) * 2, dtype=torch.float32) % 31).reshape(*shape, 2)
                        x = x.to(input_dtype)[..., 0]
                        inputs.append(x if noncontiguous else x.contiguous())
                    for chunks in (1, 2, 4, 8):
                        width = sum(math.ceil(x.shape[0] / chunks) * math.prod(x.shape[1:]) for x in inputs)
                        expected = torch.empty((chunks, width), dtype=output_dtype)
                        # Always use the native CPU implementation as the reference,
                        # including for NPU checks: never launch the failing NPU kernel.
                        original_chunk_cat(inputs, 0, chunks, out=expected)
                        actual = torch.full((chunks * width,), float("nan"), dtype=output_dtype, device=args.device)
                        ptr = actual.data_ptr()
                        device_inputs = [
                            torch.stack((x, x), dim=-1).to(args.device)[..., 0] if noncontiguous else x.to(args.device)
                            for x in inputs
                        ]
                        result = _fsdp_collectives.foreach_reduce_scatter_copy_in(device_inputs, actual, chunks)
                        assert result is None
                        assert actual.data_ptr() == ptr
                        assert actual.dtype == output_dtype
                        torch.testing.assert_close(actual.cpu().view(chunks, width), expected, rtol=0, atol=0)
                        checked += 1
        assert torch._chunk_cat is original_chunk_cat
    finally:
        _fsdp_collectives.foreach_reduce_scatter_copy_in = original
    print(f"PASS: {checked} packing cases on {args.device}; exact values, dtype, buffer identity, and None return")


if __name__ == "__main__":
    main()
