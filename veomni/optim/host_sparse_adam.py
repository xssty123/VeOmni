"""SparseAdam over the owner-local CPU rows of persistent PLE DTensors."""

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim._functional import sparse_adam

from ..distributed.host_embedding import drain_row_gradients, local_row_leaf


class HostSparseAdam(Optimizer):
    """Expose distributed moments to DCP and apply sparse updates to local storage."""

    def __init__(self, groups, lr, betas, eps):
        super().__init__(groups, dict(lr=lr, betas=betas, eps=eps))
        for group in self.param_groups:
            group["weight_decay"] = 0.0
            for weight in group["params"]:
                # Fresh DCP load needs destinations even for never-visited shards.
                self.state[weight].update(
                    step=torch.zeros((), dtype=torch.int64),
                    exp_avg=torch.zeros_like(weight),
                    exp_avg_sq=torch.zeros_like(weight),
                )
                local_row_leaf(weight)

    @torch.no_grad()
    def step(self, closure=None):
        """Update touched rows. Match each global table's Adam clock across its owners."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        weights = [w for group in self.param_groups for w in group["params"]]
        leaves = [local_row_leaf(w) for w in weights]
        for leaf in leaves:
            drain_row_gradients(leaf)
        active = torch.tensor(
            [leaf.grad is not None and leaf.grad._nnz() > 0 for leaf in leaves],
            dtype=torch.int32,
        )
        mesh = weights[0].device_mesh
        for dim in range(mesh.ndim):
            if mesh.size(dim) > 1:
                dist.all_reduce(active, op=dist.ReduceOp.MAX, group=mesh.get_group(dim))
        visited = iter(active.tolist())
        for group in self.param_groups:
            for weight in group["params"]:
                if not next(visited):
                    continue
                state = self.state[weight]
                state["step"].add_(1)
                local = local_row_leaf(weight)
                if local.grad is not None:
                    sparse_adam(
                        [local],
                        [local.grad],
                        [state["exp_avg"].to_local()],
                        [state["exp_avg_sq"].to_local()],
                        [int(state["step"])],
                        lr=group["lr"],
                        beta1=group["betas"][0],
                        beta2=group["betas"][1],
                        eps=group["eps"],
                        maximize=False,
                    )

        return loss

    def gradient_values(self):
        """Coalesce duplicate row contributions before measuring or scaling their norm."""
        for group in self.param_groups:
            for weight in group["params"]:
                local = local_row_leaf(weight)
                drain_row_gradients(local)
                if local.grad is not None:
                    local.grad = local.grad.coalesce()
                    if local.grad._nnz():
                        yield local.grad.values()

    def zero_grad(self, set_to_none=True):
        """Clear the owner-local row ledger at the ordinary gradient accumulation boundary."""
        super().zero_grad(set_to_none=set_to_none)
        for group in self.param_groups:
            for weight in group["params"]:
                leaf = local_row_leaf(weight)
                leaf._pending_host_rows.clear()
                leaf.grad = None
