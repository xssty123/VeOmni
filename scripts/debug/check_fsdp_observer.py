"""Manual CPU-only checks for the private FSDP observation entry point."""

import gc
import importlib.util
import json
import os
import sys
import tempfile
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


spec = importlib.util.spec_from_file_location(
    "fsdp_observe", Path(__file__).resolve().parents[2] / "tasks/train_vlm_fsdp_observe.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def check_optimizer_modes():
    # Match a direct tasks/*.py invocation so the existing scalar helper resolves.
    with patch.object(sys, "path", [str(Path(__file__).resolve().parents[2] / "tasks"), *sys.path]):
        for mode, expected_fused in [("scalar", False), ("native_fused", True)]:
            factory = SimpleNamespace(AdamW=torch.optim.AdamW)
            with patch.dict(os.environ, {"VEOMNI_ADAMW_SCALAR_DIAG": "0"}):
                verified = module.configure_optimizer(factory, mode)
            if expected_fused:
                assert verified is None and factory.AdamW is torch.optim.AdamW
            optimizer = factory.AdamW([torch.nn.Parameter(torch.ones(2))], foreach=False, fused=True)
            if not expected_fused:
                assert len(verified) == 1
            with tempfile.TemporaryDirectory() as directory:
                observer = module.FSDPObserver(directory)
                try:
                    observer.attach_optimizer(optimizer, expected_fused=not expected_fused)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Mismatched actual optimizer flags were accepted")
                observer.attach_optimizer(optimizer, expected_fused=expected_fused)
                observer.file.close()
                row = json.loads(observer.path.read_text())
                assert row["foreach"] is False and row["fused"] is expected_fused
                assert not optimizer.state, "Mode verification must not perform an optimizer update"
        with patch.dict(os.environ, {"VEOMNI_ADAMW_SCALAR_DIAG": "1"}):
            try:
                module.configure_optimizer(SimpleNamespace(AdamW=torch.optim.AdamW), "native_fused")
            except RuntimeError:
                pass
            else:
                raise AssertionError("Conflicting scalar override was accepted")
        try:
            module.configure_optimizer(SimpleNamespace(AdamW=torch.optim.AdamW), "typo")
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown optimizer mode was accepted")


def check():
    from veomni.trainer.vlm_trainer import VeOmniVLMArguments

    check_optimizer_modes()
    # Use the real argument class/property without model or device construction.
    for computed, configured, requested, expected in [(306, 500, 3, 3), (2, 500, 3, 2), (306, 1, 3, 1)]:
        args = object.__new__(VeOmniVLMArguments)
        args.train = SimpleNamespace(max_steps=configured, num_train_epochs=2)
        args._train_steps = computed
        try:
            args.train_steps = requested
        except AttributeError:
            pass
        else:
            raise AssertionError("Regression no longer exercises the read-only train_steps property")
        module.limit_training_loop(args, requested)
        assert args.train_steps == expected and args.train.num_train_epochs == 1
        assert args._train_steps == computed

    with tempfile.TemporaryDirectory() as directory:
        observer = module.FSDPObserver(directory)
        sentinel = object()
        failure = RuntimeError("injected original copy failure")
        fail = False
        copy_calls = 0

        def allocate():
            pass

        def copy_in(unsharded_grads, reduce_scatter_input, world_size):
            nonlocal copy_calls
            copy_calls += 1
            if fail:
                raise failure
            reduce_scatter_input.copy_(torch.cat([grad.reshape(-1) for grad in unsharded_grads]))
            return sentinel

        def reduce(fsdp_params, unsharded_grads, reduce_scatter_group, reduce_scatter_comm):
            output = torch.empty(sum(grad.numel() for grad in unsharded_grads))
            assert collectives.foreach_reduce_scatter_copy_in(unsharded_grads, output, 1) is sentinel
            unsharded_grads.clear()
            return output, sentinel

        collectives = SimpleNamespace(foreach_reduce=reduce, foreach_reduce_scatter_copy_in=copy_in)
        param_group = SimpleNamespace(foreach_reduce=reduce)
        observer.install(collectives, param_group)
        assert collectives.foreach_reduce is param_group.foreach_reduce
        params = [SimpleNamespace(_param_fqn="model.layers.1.mlp.experts.gate_up_proj")]
        grads = [torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)]
        ref = weakref.ref(grads[0])
        result = param_group.foreach_reduce(params, grads, None, SimpleNamespace(allocate=allocate))
        torch.testing.assert_close(result[0], torch.arange(12, dtype=torch.float32))
        assert result[1] is sentinel and grads == [] and copy_calls == 1
        gc.collect()
        assert ref() is None, "Observation retained an input tensor after native list clearing"
        assert observer.context.get() is None

        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW([parameter], foreach=False, fused=False)
        observer.attach_optimizer(optimizer)
        for _ in range(2):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        assert observer.updates == 2
        fail = True
        try:
            param_group.foreach_reduce(params, [torch.ones(2)], None, SimpleNamespace(allocate=allocate))
        except RuntimeError as error:
            assert error is failure, "Original exception identity changed"
        else:
            raise AssertionError("Original failure was swallowed")
        assert observer.context.get() is None and copy_calls == 2
        observer.file.close()
        rows = [json.loads(line) for line in observer.path.read_text().splitlines()]
        assert sum(row["event"] == "optimizer_return" for row in rows) == 2
        assert sum(row["event"] == "copy_exception" for row in rows) == 1
        assert sum(row["event"] == "reduce_exception" for row in rows) == 1
        assert rows[1]["parameter_fqns"] == [params[0]._param_fqn]
        assert rows[1]["reduce_scatter_world_size"] == 1
        assert rows[-1]["completed_optimizer_steps"] == 2
    print(
        "PASS: real argument loop cap, CPU values, aliases, return/exception identity, tensor release, updates, JSON"
    )


if __name__ == "__main__":
    check()
