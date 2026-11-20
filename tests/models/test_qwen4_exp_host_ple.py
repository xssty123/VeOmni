"""Two-rank regression for host PLE lookup, sparse updates, clipping and DCP resume."""

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from transformers import Qwen4ExpConfig

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.test_qwen4_exp_packing import _forward_varlen
from tests.tools.launch_utils import spawn_with_rendezvous
from tests.tools.training_utils import make_eager_ops_config
from veomni.distributed.fsdp2.clip_grad_norm import extra_parallel_fsdp2_clip_grad_norm
from veomni.distributed.host_embedding import HostEmbedding, local_row_leaf
from veomni.distributed.parallel_state import _init_parallel_state
from veomni.models.configuration_base import as_veomni_config
from veomni.models.registry import get_model_config
from veomni.models.transformers.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel, Qwen4ExpTextNGramEmbedding
from veomni.models.transformers.qwen4_exp.parallel_plan import get_parallel_plan
from veomni.optim.optimizer import build_optimizer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))


def _build(ps):
    config = Qwen4ExpConfig.from_pretrained("tests/toy_config/qwen4_exp_toy/config.json").text_config
    config.ple_host_offload = True
    root = nn.Module()
    layer = nn.Module()
    layer.ple = nn.Module()
    embed = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, 0)
    layer.ple.ple_embedding = embed
    root.model = nn.Module()
    root.model.language_model = nn.Module()
    root.model.language_model.layers = nn.ModuleList([layer])
    get_parallel_plan().apply(root, ps.extra_parallel_fsdp_device_mesh)
    opt = build_optimizer(root, lr=0.01, weight_decay=0)
    return root, embed, opt


def _compare_packed_text_stack(ps, rank):
    outputs = []
    gradients = []
    for host in (False, True):
        config = get_model_config(
            "tests/toy_config/qwen4_exp_toy",
            text_config={
                "ple_host_offload": host,
                "ple_layer_ids": [1, 2],
                "num_hidden_layers": 3,
                "layer_types": ["linear_attention", "linear_attention", "full_attention"],
            },
        ).text_config
        as_veomni_config(config, validate=False)._set_ops_implementation(make_eager_ops_config())
        torch.manual_seed(109)
        root = nn.Module()
        root.model = nn.Module()
        text = Qwen4ExpTextModel(config)
        root.model.language_model = text
        get_parallel_plan().apply(root, ps.extra_parallel_fsdp_device_mesh)
        text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ids = torch.tensor([[11, 11, 12, 13, 15, 15]]) + rank
        y = _forward_varlen(text, ids, (2, 4))
        outputs.append(y.detach())
        y.square().sum().backward()
        weights = [
            table.weight
            for layer in text.layers
            if layer.ple is not None
            for table in layer.ple.ple_embedding.ngram_embedding.values()
        ]
        row_grads = []
        for weight in weights:
            grad = local_row_leaf(weight).grad if host else weight.grad
            if grad is None:
                row_grads.append(torch.zeros_like(weight.to_local()))
            else:
                row_grads.append(grad.to_dense() if host else grad.to_local())
        gradients.append(row_grads)
    torch.testing.assert_close(*outputs)
    for dense, sparse in zip(*gradients, strict=True):
        torch.testing.assert_close(dense, sparse)


def _worker(rank, world_size, path, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=world_size)
    try:
        ps = _init_parallel_state(
            dp_size=world_size,
            dp_shard_size=world_size,
            device_type="cpu",
            extra_parallel_sizes=(world_size,),
            extra_parallel_names=("ple",),
        )
        # Production casts the meta model, shards tables, then materializes
        # only each owner's slice. The table must never allocate the full shape.
        with torch.device("meta"):
            meta_table = HostEmbedding(16, 4, dtype=torch.float32)
        from torch.distributed.tensor import DTensor, Shard

        layout = ps.extra_parallel_fsdp_device_mesh["ple"]
        source = torch.empty(8, 4, device="meta")
        meta_table.weight = nn.Parameter(DTensor.from_local(source, layout, [Shard(1), Shard(0)]))
        meta_table.float()
        assert meta_table.weight.is_meta
        meta_table.to_empty(device="cpu")
        assert meta_table.weight.shape == (16, 4)
        assert meta_table.weight.to_local().shape == (8, 4)
        assert meta_table.weight.to_local().device.type == "cpu"
        _compare_packed_text_stack(ps, rank)
        torch.manual_seed(43)
        model, embed, opt = _build(ps)
        table = embed.ngram_embedding["shard_0"]
        weight = table.weight
        ref = nn.Embedding.from_pretrained(weight.full_tensor().detach().clone(), freeze=False, sparse=True)
        ref_opt = torch.optim.SparseAdam(ref.parameters(), lr=0.01, betas=(0.9, 0.95))
        # Step zero sends all requests to a single owner. Step one crosses the mesh.
        for step in range(2):
            ids = torch.tensor([rank, 0, 0 if step == 0 else weight.size(0) - 1])
            rows = embed._distributed_lookup(torch.zeros_like(ids), ids)
            torch.testing.assert_close(rows, ref(ids))
            rows.sum().backward()
            all_ids = [torch.empty_like(ids) for _ in range(world_size)]
            dist.all_gather(all_ids, ids)
            (ref(torch.cat(all_ids)).sum() / ps.extra_parallel_gradient_divide_factor("ple")).backward()
            ref_grad = ref.weight.grad.coalesce()
            norm = ref_grad.values().norm()
            torch.testing.assert_close(extra_parallel_fsdp2_clip_grad_norm(model, 0.7), norm)
            ref_grad.values().mul_((0.7 / (norm + 1e-6)).clamp(max=1))
            ref.weight.grad = ref_grad
            opt.step()
            ref_opt.step()
            torch.testing.assert_close(weight.full_tensor(), ref.weight)
            opt.zero_grad()
            ref_opt.zero_grad()
            assert local_row_leaf(weight).grad is None
            if step == 0:
                dcp.save({"model": model.state_dict(), "opt": opt.state_dict()}, checkpoint_id=path)
                model, embed, opt = _build(ps)
                state = {"model": model.state_dict(), "opt": opt.state_dict()}
                dcp.load(state, checkpoint_id=path)
                model.load_state_dict(state["model"])
                opt.load_state_dict(state["opt"])
                weight = embed.ngram_embedding["shard_0"].weight
                restored = model._host_ple_optimizer.state[weight]
                assert int(restored["step"]) == 1
                for name in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(restored[name].full_tensor(), ref_opt.state[ref.weight][name])
    finally:
        dist.destroy_process_group()


def test_host_ple_against_sparse_adam_and_resume(tmp_path):
    spawn_with_rendezvous(_worker, args=(2, str(tmp_path)), nprocs=2)


def test_host_ple_materializes_fp32_on_cpu():
    with torch.device("meta"):
        table = HostEmbedding(8, 4, dtype=torch.float32)
    table.float()  # build_parallelize_model casts the model before applying its parallel plan.
    assert table.weight.is_meta
    table.to_empty(device="cpu")
    table.to(dtype=torch.bfloat16)
    assert table.weight.device.type == "cpu" and table.weight.dtype == torch.float32


def test_deferred_rows_wait_and_accumulate():
    from veomni.distributed.host_embedding import drain_row_gradients

    leaf = local_row_leaf(HostEmbedding(4, 2, dtype=torch.float32).weight)
    values = torch.zeros(2, 2, dtype=torch.bfloat16)

    class CopyEvent:
        def synchronize(self):
            values.fill_(2)

    nn.functional.embedding(torch.tensor([1]), leaf, sparse=True).sum().backward()
    leaf._pending_host_rows.append((CopyEvent(), torch.tensor([1, 2]), values))
    drain_row_gradients(leaf)
    torch.testing.assert_close(leaf.grad.to_dense(), torch.tensor([[0, 0], [3, 3], [2, 2], [0, 0.0]]))
    assert not leaf._pending_host_rows


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=1, device="npu"))
def test_native_host_row_transfers():
    from veomni.distributed.host_embedding import drain_row_gradients, lookup_host_rows
    from veomni.utils.device import IS_NPU_AVAILABLE

    if not IS_NPU_AVAILABLE:
        pytest.skip("requires Ascend pinned memory and copy streams")
    tables = nn.ModuleDict({"shard_0": HostEmbedding(4, 2, dtype=torch.float32)})
    leaf = local_row_leaf(tables["shard_0"].weight)
    ids = torch.tensor([2, 1, 2], device="npu")
    rows = lookup_host_rows(tables, torch.zeros_like(ids), ids, torch.bfloat16, 4)
    torch.testing.assert_close(rows.cpu(), leaf.detach()[[2, 1, 2]].bfloat16())
    rows.float().sum().backward()
    assert leaf._pending_host_rows
    drain_row_gradients(leaf)
    torch.testing.assert_close(leaf.grad.to_dense(), torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]))
