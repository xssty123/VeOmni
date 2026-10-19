# Ascend A5 ChunkCat diagnostic branch

## Native FSDP observation baseline

`tasks/train_vlm_fsdp_observe.py` is a separate, opt-in torch 2.10 entry.
It calls the installed native `foreach_reduce` and copy-in functions unchanged,
records parameter FQNs in gradient-list order, reduction group size, allocation
binding, stream representation, input/output metadata, and optimizer entry/return.
It does **not** install eager packing or fresh takeover, add synchronization,
or retain tensors/events between calls. Returning from a function is a host
boundary, **not** proof of device completion. Logging can still affect host timing.
This first observer maps groups and update boundaries; it does not yet trace
allocator internals, event waits, or reference release.

Each rank writes a separate JSONL file under `VEOMNI_FSDP_OBSERVE_DIR`, capped
at 32 MiB with a `trace_limit` marker. The previous 8 MiB cap covered three
steps (about 5.2 MB per rank); 32 MiB provides headroom for a ten-step follow-up.
There is no per-copy-row output. Parameter
names are recorded once per group; `group_id` connects later calls to that list.
The entry requires a fresh run. `VEOMNI_FSDP_OBSERVE_OPTIMIZER=scalar` (default)
creates/verifies AdamW False/False using the existing scalar diagnostic helper.
For a single-variable fused comparison, set it to `native_fused` and set
`VEOMNI_ADAMW_SCALAR_DIAG=0`: the repository's optimizer factory is left intact,
and every actual AdamW group must have `foreach=False, fused=True` before the
training loop. A mismatch aborts rather than silently testing the wrong path.
Keep reuse, blocking, topology, data, observation and step limit at baseline.
The ordinary training entry and the previous takeover entry remain unchanged.

`VEOMNI_FSDP_OBSERVE_STEPS` defaults to 3. After normal trainer and scheduler
construction, only the training loop is capped to that many steps in one epoch;
the already-built LR scheduler keeps the original horizon. `loop_cap` records
both horizons. `completed_optimizer_steps` counts returned optimizer calls,
not independently verified finite gradients; also inspect normal loss/grad logs.
Switching from the takeover diagnostic to this entry rebuilds a **native
baseline**, and is not a single-variable topology experiment. Keep EP/SP fixed
for this run before comparing topology changes. Do not compare raw call numbers
across different parallel layouts.

Manual CPU checks (no training or accelerator computation):

```bash
python scripts/debug/check_fsdp_observer.py
```

These check value/return/exception preservation, alias wiring, input tensor
release, per-rank records, and optimizer counters. They do not validate NPU
execution or the private FSDP fields on a live model.

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

The check exercises 144 cases per packing mode (576 total, including expected layout rejections) with native CPU ChunkCat, covering padding,
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

Look for `[chunkcat-eager] installed` on every rank with `boundary_sync=1`
and the selected `packing_mode` (`schema=7` in this revision).
The debug entry now defaults `VEOMNI_CHUNKCAT_SYNC` to `1`: device-wide NPU
synchronization immediately before and after each packing call. This changes
timing and overlap, so a passing run is diagnostic evidence, not a validated fix.
Set `VEOMNI_CHUNKCAT_SYNC=0` explicitly to compare without these added barriers.
The ordinary training entry is unaffected. No distributed barrier is added.

Events contain a JSON record after the `[chunkcat-eager] EVENT` prefix, including
rank, PID, call number, stage, and timestamp. Per-call metadata records include
input/output shapes, strides, dtype, device, data pointers, storage offsets,
logical byte sizes, and the current stream representation. Stream handles in
Python must not be assumed to equal the device log's numeric stream IDs.
Metadata is cached before packing; failures print only cached metadata and the
exception, then re-raise without further device queries or synchronization.

| Failure stage | Interpretation |
| --- | --- |
| `metadata` / `stream_context` | Diagnostic setup failed; packing has not started. |
| `pre_sync` | Device work failed before this packing call was issued. |
| `pack` | The eager decomposition raised after pre-sync succeeded. |
| `post_sync` | Packing returned, but subsequent device synchronization failed. |

Device synchronization covers all streams, so these boundaries localize error
reporting, not necessarily the original corruption. Other submitted work and
earlier undetected corruption are not ruled out by successful pre-sync.
`submitted` alone does not establish device completion. With synchronization
enabled, expect `complete` at `post_sync` before returning to FSDP. With it off,
`VEOMNI_CHUNKCAT_LOG_INPUTS=1` keeps per-call metadata; otherwise only the first
call and failures are logged. Do not infer finite gradients from a printed step.
The CPU check also injects pre-sync, packing, and post-sync failures to verify
stage attribution, original exception propagation, and no subsequent operations.

### Split packing inside training

Set `VEOMNI_CHUNKCAT_MODE=precast` for the next training diagnostic, keeping
`VEOMNI_CHUNKCAT_SYNC=1`. The entry still uses the real FSDP gradients, output
buffer, and current stream, but splits the work into synchronized stages:

1. `pad_and_reshape`: PyTorch 2.10 `_pad_chunk` on the entire tensor list.
2. `cast_tensor_0`, `cast_tensor_1`, etc.: convert each packed tensor to the output
   dtype (FP32 in the failing training run), recording the zero-based index and
   converted buffer metadata. Same-dtype tensors may alias instead of copying.
3. `same_dtype_cat`: concatenate those tensors into the original output buffer.

Each operation logs `begin`, `submitted`, then `complete` after synchronization.
A failure before `submitted` occurred in the operation; a failure after it but
before `complete` occurred at its synchronization. Cached failure metadata
includes packed inputs and all conversions that completed before the failure.
CPU checks inject failures in padding, both example casts, concatenation, and
a cast's synchronization; each failure must stop subsequent phases.

This removes the implicit mixed-dtype conversion inside `cat`, so it is a
diagnostic comparison, not an identical kernel sequence. Converted tensors are
retained until concatenation: the observed 15-input group needs approximately
3.35 GB of additional FP32 storage, plus any padding intermediates. The changed
allocation and synchronization can affect reproduction. Do not interpret an OOM
in this mode as the original Cast fault, or a pass as proof of a root-cause fix.
The entry defaults to `eager`; set `VEOMNI_CHUNKCAT_MODE=eager` to restore the
previous packing path while retaining boundary diagnostics. The server's
experiment launcher may explicitly select `precast`; inspect its effective log.

### Row-copy comparison after same-dtype concat failure

The precast training run reached call 545 on all eight ranks: all 15 casts and
synchronizations completed, then `same_dtype_cat` failed. Device logs identified
ConcatD. This localizes the failure boundary, not the underlying cause.

Set `VEOMNI_CHUNKCAT_MODE=rowcopy` with `VEOMNI_CHUNKCAT_SYNC=1` to retain the
padding, all converted tensors, and the original FSDP output buffer, but replace
`torch.cat` with sequential row copies. Each converted tensor has shape
`(world_size, width)`. For each tensor and row, copy into that output row's
corresponding column interval. Both source and destination slices are contiguous;
no new full-size output is allocated. This preserves rank-major packing and the
output buffer identity. The 15-tensor, eight-row group performs 120 copies.

Stages `copy_tensor_I_row_R` log source/destination shape, stride, dtype, address,
storage offset, byte size, and destination column bounds before copying. Each
copy has its own device synchronization. On failure, the cached `copy` field
identifies the active interval; no further device work is attempted. Layout
validation rejects incompatible dimensions/widths or noncontiguous buffers.
The entry logs `schema=6`. The experiment launcher may select `rowcopy` by default;
check `packing_mode` in the installed message and saved runtime environment.

This changes kernel selection, transfer sizes, timing, and synchronization count.
A pass supports a usable diagnostic alternative, not proof of a ConcatD bug or
absence of prior memory corruption. Retained FP32 temporary storage is unchanged
from precast (about 3.35 GB for the failing group). Performance is not evaluated.
Same-dtype conversions can retain noncontiguous input layouts; those rowcopy
cases are expected to fail layout validation before any output writes. The check
reports successful parity cases and rejected-layout cases separately.
CPU checks forbid cat in rowcopy, compare exact rank-major values, check output
guards with a nonzero storage offset, and inject early/late copy and sync failures.

### Fresh destination probe at one packing call

The rowcopy training run also failed at call 545 after all 15 casts completed.
The failing device kernel was TensorMove, so bypassing ConcatD did not resolve
the observed failure. This does not establish whether the source, destination,
allocation lifecycle, earlier corruption, or device runtime caused the error.

Set `VEOMNI_CHUNKCAT_MODE=rowcopy_fresh`, `VEOMNI_CHUNKCAT_SYNC=1`, and
`VEOMNI_CHUNKCAT_PROBE_CALL=545` (the default probe call). All other calls retain
rowcopy. At the selected call, after layout validation and all casts:

1. `allocate_fresh_output` allocates and synchronizes an `empty_like` output while
   the original FSDP output remains alive. Both buffer addresses are recorded;
   identical live addresses are rejected.
2. `fresh_copy_tensor_I_row_R` copies each converted source slice to the fresh
   destination, using the same order, copy sizes, and per-copy synchronization.
3. Only after all fresh copies complete, `copy_tensor_I_row_R` replays the same
   **converted sources** into the **original FSDP output**. This is not a copy
   from the fresh buffer back to the original. Both outputs stay alive throughout
   the comparison; successful return preserves the original output identity and
   the FSDP helper's `None` return contract.

Each completed target emits `target_complete` with `target=fresh|original`.
Failure immediately stops further work and reports cached `out`, `fresh_out`
(if allocated and synchronized), and active `copy.target` metadata. A fresh-copy
failure leaves the original output unwritten; it cannot tell us whether an
original-buffer copy would have passed afterward. A fresh pass followed by an
original failure strengthens the destination-specific hypothesis but does not
prove it: the extra allocation, elapsed time, and repeated reads change runtime
conditions. Two passes mean this perturbed call passed, not that the original
failure is fixed. Allocation failure/OOM is a separate diagnostic outcome.

The probe adds another output-sized allocation (about 3.35 GB / 3.12 GiB for the
observed group), in addition to precast's FP32 temporaries. No new destination is
allocated on other calls. Inspect the actual call's input/output metadata: a call
number is not a model-layer identity and can change if the workload changes.
CPU checks cover selected/nonselected calls, identical source pointers and copy
sizes across both passes, output guards, precise values, and injected allocation,
copy, and synchronization failures. They do not validate A5 memory behavior.

### Allocation provenance at the selected call

Schema 6 adds observation only to the selected `rowcopy_fresh` call. The
`allocation_context` event inspects the direct `foreach_reduce` caller and checks
that its `reduce_scatter_input` is the same object. It records the actual
communication class, the bound `allocate` implementation's module, qualified
name, source file and line, and boolean instance flags `_use_mem_cache` and
`_mem_cache_flag`. Missing flags are null; an unrecognized caller is reported
as unavailable. No allocator method is invoked or replaced by this observation.

These flags are sampled **after allocation**, at copy-in. In particular, a true
`_mem_cache_flag` alone does not prove which branch served this allocation.
Use the recorded implementation and its current source to interpret the flags;
the presence of a torch_npu patch on disk does not prove that it was activated.

`out_storage` and `fresh_storage` record storage base address, byte capacity,
tensor address, contiguous logical end, and whether that interval fits inside
the storage. Noncontiguous bounds are left unknown. These are PyTorch-visible
metadata and device virtual addresses, not proof of valid physical backing,
allocation lifetime, or absence of earlier corruption. Snapshots are captured
before copies and reused in failure reports without additional device queries.
Copy order, temporary allocations, synchronization, and the FSDP output remain
unchanged from the preceding fresh-output experiment. CPU checks cover allocator
provenance without calling it, offset views, noncontiguous bounds, missing caller
fallback, and preservation of cached metadata on injected failures.

### Independent size probes (schema 7)

Keep `rowcopy_fresh`, synchronization, and the selected call fixed. Both controls
are disabled by default and are mutually exclusive; setting both aborts before
training setup. Ordinary training and all nonselected packing calls are unchanged.

* `VEOMNI_CHUNKCAT_COPY_CHUNK_BYTES=8388608`: split only the selected original
  row into at most 8 MiB per copy, with synchronization after every part. The
  default row is tensor 7 / row 5 for ranks 0–3, tensor 8 / row 4 for ranks 4–7.
  Other ranks require explicit `VEOMNI_CHUNKCAT_COPY_SLICE=INDEX:ROW`. The latter
  can also override the slice for a different reproduction. Indices are zero
  based. Fresh copies and all preceding original copies retain their old sizes.
  The entire row is covered, including a short final part. `copy_part` records
  source/destination addresses and row byte offsets; `split_row_complete` means
  all parts synchronized. A failure aborts without a larger-copy retry. This
  changes copy size, kernel dispatch, timing, and synchronization count, but not
  the logical data, original allocation, or row coverage. Compare actual base
  addresses across runs; do not reuse a numeric address from an exited process.
* `VEOMNI_CHUNKCAT_ALLOC_EXTRA_BYTES=67108864`: enlarge the original FSDP
  allocation at the selected call by 64 MiB, returning its zero-offset prefix
  with the original shape/dtype/stride. Only `DefaultReduceScatter.allocate` is
  wrapped, and only when called directly by the inspected `foreach_reduce`
  using that communication object. The unpatched `DefaultAllocMixin.allocate`
  signature and single `torch.empty` implementation are checked before wrapping.
  Caller matching unwraps `foreach_reduce`'s `no_grad` decorator to compare
  against the function body that actually calls the allocator.
  The same original allocator allocates the larger storage; no original-sized
  throwaway allocation is made. All-gather and nonselected allocations are
  untouched. The returned view keeps the backing storage alive naturally; the
  diagnostic stores a weak reference and verifies identity/capacity at copy-in.
  A bypassed/replaced hook aborts instead of silently testing the old capacity.
  Copy sizes and fresh-buffer size remain unchanged. Additional capacity can
  change addresses, memory pressure, mapping and subsequent fresh allocation;
  passing is not proof of a capacity-only root cause. Requested storage bytes
  are not necessarily the underlying allocator block's reserved/physical size.

Allocation events are `allocation_resize_begin`, `allocation_resize_returned`,
and `allocation_resize_failed`. Returned does not mean device completion; the
existing packing pre-sync still provides the device boundary. These events
occur before copy-in and do not carry its usual stage/event JSON fields.
Never enable either experiment together with another allocator change. Do not
shrink the actual SFT output below its required logical capacity. Smaller
standalone allocations remain a separate follow-up, not a training equivalence
check. Each experiment runs in a new process and must be examined before the
next; no automatic size sweep or training launch is provided.

CPU validation (no NPU allocation or training):

```bash
python scripts/debug/check_chunkcat_sizes.py
python scripts/debug/check_chunkcat_eager.py --device cpu
```

The size checks cover decorated callers and the installed PyTorch
`foreach_reduce` allocation/copy/output path on small CPU tensors. Device-stream
plumbing is stubbed, and execution stops before communication. They verify the
selected input grows by 64 bytes or 64 MiB while neighboring calls and the
communication output keep their original capacities. This does not validate
NPU allocation or distributed training.

Rollback: run `tasks/train_vlm.py` in a new process. Do not change AdamW foreach
at the same time. Full logs, device dumps, datasets, model weights, host paths,
and credentials remain outside this public fork.

## KKT numerical comparison with the `npu` GDN backend

The `npu` backend's `triton/chunk_scaled_dot_kkt.py` forward kernel now follows
the supplied KKT patch: an explicit strict-lower mask includes valid rows and
columns, each dot tile is masked with `where` before accumulation, gating uses
`exp(clamp(g_i - g_j, -50, 50))`, beta scales the rows, and a final `where`
clears masked lanes again after scaling. The patch's chunk/token and beta names
map to this kernel's `i_t`, `T_local`, and `b_beta`; indexing and launch geometry
remain unchanged. The `npu_ascendc` backend's separate `triton_core` copy is not
part of this numerical comparison. This change does not replace the FSDP
original output with fresh storage or establish an A5 numerical pass.

## Standalone Cast / packing reproduction

The first eager SFT attempt reached step 1 with finite logged gradient norm, then
failed at packing call 545. All ranks reached the decomposition's `torch.cat`;
device logs reported a Cast kernel instead of the earlier ChunkCat kernel.
`scripts/debug/reproduce_a5_cast.py` captures the 15 input shapes and strides,
BF16 inputs, FP32 output shape `(8, 104532544)`, and eight packing chunks.
This is a single-device test: eight chunks do not require eight NPUs or HCCL.

The largest input has 536870912 elements (2 GiB after FP32 conversion). The
observed failure does not establish a size-limit bug: matching metadata had
already appeared repeatedly earlier in training. Synthetic per-tensor constants
are used here, not captured gradient values, addresses, or training stream state.
A standalone pass cannot rule out a training-context problem.

From the existing checkout, inside the prepared container, run manually:

```bash
# Default: npu:0, real shapes, 100 repetitions, reused input/output buffers.
bash scripts/debug/run_a5_cast_repro.sh eager

# Run these as separate experiments after inspecting any failure:
bash scripts/debug/run_a5_cast_repro.sh cast
bash scripts/debug/run_a5_cast_repro.sh precast-cat
bash scripts/debug/run_a5_cast_repro.sh mixed-cat

# Allocation comparison; the allocator may return previously used addresses.
bash scripts/debug/run_a5_cast_repro.sh eager --allocation fresh

# Isolate the largest tensor's conversion (index 7, zero based).
bash scripts/debug/run_a5_cast_repro.sh cast --tensor-index 7
```

| Mode | Operations |
| --- | --- |
| `eager` | Call the same PyTorch decomposition used by the debug training entry. |
| `cast` | Pad/reshape, then convert each BF16 tensor to FP32 separately. |
| `precast-cat` | Pad/reshape, convert each tensor explicitly, then concatenate FP32 tensors into FP32 output. |
| `mixed-cat` | Pad/reshape, then concatenate BF16 tensors directly into FP32 output. |

All paths synchronize after each named phase and check deterministic samples,
including padding. `mixed-cat` adds a synchronization boundary after padding,
while `eager` preserves the decomposition as one Python call. Cast mode releases
each converted tensor after checking; precast-cat keeps all converted inputs
until concatenation. These allocation differences are intentional and logged.
The tests use the default stream and synthetic data; they are not full training
equivalence or performance tests. Real cases need several GiB of device memory.

The launcher creates a unique directory under `../diagnostics/` containing full
stdout/stderr, exit codes, JSONL phase events, commit/diff, and script snapshots.
`begin` without `complete`, or a `failed` event, identifies the phase to inspect;
`complete` is emitted only after synchronization. `all` runs each mode in a fresh
process and stops on the first failure. No package installation or training is
performed. Choose another device with `REPRO_DEVICE=npu:1`; existing visibility
settings still apply. The launcher sets `ASCEND_LAUNCH_BLOCKING=1` and defaults
`MULTI_STREAM_MEMORY_REUSE` to 2, recording their effective values.

CPU plumbing checks (small shapes only, not A5 validation):

```bash
REPRO_DEVICE=cpu bash scripts/debug/run_a5_cast_repro.sh all --size tiny --iterations 2
REPRO_DEVICE=cpu bash scripts/debug/run_a5_cast_repro.sh all --size tiny --iterations 2 --allocation fresh
```

## Single-call fresh input takeover

`VEOMNI_CHUNKCAT_MODE=rowcopy_takeover` preserves the synchronous row-copy
diagnostic, but at `VEOMNI_CHUNKCAT_PROBE_CALL` (default 545) fills only the fresh
buffer. Both size controls must be zero and `VEOMNI_CHUNKCAT_SYNC=1` is required.
The original buffer stays alive while fresh is allocated and filled, as in the
comparison probe. After copying, the original buffer can be released normally.

The entry adapts the installed torch 2.10 `foreach_reduce` in process: its
copy-in call binds the returned tensor to `reduce_scatter_input`. Signature,
decorator, copy site, communication input and returned lifetime-state structure
are checked before installation. Both the collectives and parameter-group aliases
are replaced; installed torch files remain untouched. Stream waits and returned
communication events are preserved. Other calls still fill and return original.

At the selected call, `takeover_selected`, `communication_input` and
`lifetime_input` report the fresh pointer at selection, communication submission
and return. Submission is not device completion. No original row copies should
appear for that call. Later calls are deliberately not switched automatically.
This is a diagnostic bypass, not a root-cause fix or numerical-equivalence claim.

`python scripts/debug/check_chunkcat_takeover.py` runs the installed FSDP function
on CPU with simulated communication/streams. It checks exact packing and output
gradients, call selection, original-buffer preservation, returned event/input
identity, reference release and fail-stop behavior. It does not validate NPU
stream execution or actual distributed communication.

## AdamW implementation comparison

With this diagnostic entry, `VEOMNI_ADAMW_SCALAR_DIAG=1` forces VeOmni's AdamW
constructor to receive `foreach=False, fused=False` before optimizer state is
created. The default `0` preserves the existing SFT fused optimizer. This does
not change FSDP packing, GDN, or installed torch files. Each constructed AdamW
checks its actual parameter groups and logs `[adamw-scalar-diag] verified` with
the rank and group count. Conflicting group overrides or an unused constructor
stop the run before training. This is a controlled comparison, not a proven fix.
