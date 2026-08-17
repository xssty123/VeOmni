from types import SimpleNamespace
from unittest.mock import MagicMock, call

from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.trace_callback import TqdmCallback


def test_step_end_updates_postfix_without_extra_refresh():
    callback = TqdmCallback.__new__(TqdmCallback)
    callback.trainer = SimpleNamespace(
        step_train_metrics={
            "training/total_loss": 11.31,
            "training/foundation_loss": 11.31,
            "training/grad_norm": 55.14,
            "training/lr": 0.0,
        }
    )
    callback.data_loader_tqdm = MagicMock()

    callback.on_step_end(TrainerState())

    assert callback.data_loader_tqdm.method_calls == [
        call.set_postfix_str(
            "total_loss: 11.31, foundation_loss: 11.31, grad_norm: 55.14, lr: 0.00",
            refresh=False,
        ),
        call.update(),
    ]
