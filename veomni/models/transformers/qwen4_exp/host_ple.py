"""Prefetch Qwen PLE lookups one decoder layer ahead outside block checkpointing."""

from functools import lru_cache

from veomni.utils.device import create_event, create_stream, get_current_stream, switch_to_specified_stream


@lru_cache(maxsize=8)
def _lookup_stream(device):
    return create_stream(device)


def prefetch_layers(layers, input_ids, dtype, cu_seq_lens, packed_positions):
    """Yield each layer and its differentiable, stream-ready PLE embeddings."""
    device = input_ids.device
    stream = None if device.type == "cpu" else _lookup_stream(device)
    main = None if stream is None else get_current_stream(device)
    if stream is not None:
        stream.wait_stream(main)
        for tensor in (input_ids, cu_seq_lens, packed_positions):
            if tensor is not None and tensor.device == device:
                tensor.record_stream(stream)

    def lookup(layer):
        if layer.ple is None:
            return None, None

        def run():
            return layer.ple.ple_embedding(
                input_ids, None, output_dtype=dtype, cu_seq_lens_q=cu_seq_lens, packed_positions=packed_positions
            )

        if stream is None:
            return run(), None
        with switch_to_specified_stream(stream):
            rows = run()
            event = create_event()
            event.record(stream)
        return rows, event

    pending = lookup(layers[0]) if layers else (None, None)
    for i, layer in enumerate(layers):
        rows, event = pending
        if i + 1 < len(layers):
            pending = lookup(layers[i + 1])
        if event is not None:
            main.wait_event(event)
            rows.record_stream(main)
        yield layer, rows
