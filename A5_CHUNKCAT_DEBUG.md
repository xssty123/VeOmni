# Ascend A5 ChunkCat diagnostic branch

This branch starts from upstream `1af5452cfc5db9e16660772e1191d48f15fc450c`.
The earlier SFT reproduction used `76f32077f42ac496c2dccfb2a35a97c337e1385a`;
moving to latest main is a separate baseline change, so results are not a
single-variable comparison with that earlier checkout.

## Preserved experimental modification

The two arch35 guards in `veomni/ops/kernels/gated_delta_rule/_ascend/causal_conv1d.py`
are commented out, matching the existing A5 experiment. This is an experimental
guard bypass, not a claim that the kernel has passed A5 correctness validation.
Dependency versions, model-generated code, optimizer settings, and HCCL settings
are not changed by this branch.

## Eager packing entry point

`tasks/train_vlm_chunkcat_eager.py` replaces only PyTorch 2.10 FSDP's
`foreach_reduce_scatter_copy_in` in each worker process. It uses the installed
`torch._decomp.decompositions._chunk_cat`: padding, reshape, and ordinary `cat`
into the existing output buffer. The original output dtype (including BF16
inputs with FP32 reduction), rank-major packing, and downstream reduce-scatter
are preserved. No files under site-packages are modified. The ordinary
`tasks/train_vlm.py` entry point retains native ChunkCat.

This does not mean disabling torch.compile: the failing experiment already used
eager execution. It replaces the dedicated ChunkCat kernel with its decomposition.
Both entry points still execute tensor operations on the selected accelerator.
Extra padding/copies can increase memory use and runtime. A passing fallback is
workaround evidence, not proof of the original kernel's root cause.

From the checkout root, check packing independently (no model or training):

```bash
python scripts/debug/check_chunkcat_eager.py --device cpu
# Optional hardware check, run manually on an available device:
python scripts/debug/check_chunkcat_eager.py --device npu:0
```

The check compares 144 cases with native CPU ChunkCat, covering padding,
noncontiguous inputs, mixed tensor ranks, BF16/FP32 output, and world sizes 1/2/4/8.
It is a manual diagnostic, not an automatically wired CI test.

To train later, use an existing path-adapted SFT YAML with a fresh checkpoint
output directory. Keep GDN `npu_ascendc` and the selected parallel settings fixed:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_LAUNCH_BLOCKING=1
export MULTI_STREAM_MEMORY_REUSE=2
export VEOMNI_CHUNKCAT_LOG_INPUTS=1
export NPROC_PER_NODE=8
bash train.sh tasks/train_vlm_chunkcat_eager.py /absolute/path/to/training.yaml
```

Look for `[chunkcat-eager] installed` on every rank and `begin` metadata from
actual calls. `submitted` means the Python call returned, not that asynchronous
device work completed. Disable per-call metadata by unsetting
`VEOMNI_CHUNKCAT_LOG_INPUTS`; the first call and Python failures are still logged.
Do not infer finite gradients from a printed step alone.

Rollback: run `tasks/train_vlm.py` in a new process. Do not change AdamW foreach
at the same time. Full logs, device dumps, datasets, model weights, host paths,
and credentials remain outside this public fork.
