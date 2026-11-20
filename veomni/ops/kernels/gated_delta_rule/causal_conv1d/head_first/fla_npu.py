# Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""AscendC training convolution with contiguous head-first output for grouped GDN."""

from itertools import accumulate

import torch.nn.functional as F

from .....registry import SavedState


def forward(x, weight, bias=None, *, heads, activation, packed_lengths=None):
    """Convolve [B,T,C] depthwise with [C,W] weights and return [B,H,T,D]."""
    from fla_npu.ops.ascendc import causal_conv1d_fn

    if activation not in (None, "silu", "swish"):
        raise ValueError("AscendC causal convolution supports SiLU or no activation")
    if heads < 1 or x.size(-1) % heads:
        raise ValueError("The input channel count must be divisible by the number of convolution heads")
    bounds = None
    if packed_lengths is not None:
        if x.size(0) != 1 or not packed_lengths or sum(packed_lengths) != x.size(1) or min(packed_lengths) <= 0:
            raise ValueError("Packed AscendC convolution needs positive lengths covering one flattened sequence")
        bounds = list(accumulate(packed_lengths, initial=0))
    op_x = (x.squeeze(0) if bounds is not None else x).contiguous()
    op_weight = weight.t().contiguous()
    # The native SiLU backward consumes the preactivation in the same head layout.
    preactivation = causal_conv1d_fn(
        op_x,
        op_weight,
        bias,
        activation=None,
        head_num=heads,
        query_start_loc_cpu=bounds,
    )
    output = F.silu(preactivation) if activation else preactivation
    if bounds is not None:
        output = output.unsqueeze(0)
    return output, SavedState((op_x, op_weight, preactivation), (bounds, bool(activation), bias is not None))


def backward(grad_output, saved):
    """Return input, filter and optional bias gradients from the AscendC training kernel."""
    from fla_npu.ops.ascendc import causal_conv1d_bwd

    op_x, op_weight, preactivation = saved.tensors
    bounds, activated, has_bias = saved.metadata
    dy = grad_output.squeeze(0) if bounds is not None else grad_output
    dx, dw, db, _ = causal_conv1d_bwd(
        x=op_x,
        y=preactivation if activated else None,
        weight=op_weight,
        dy=dy.contiguous(),
        query_start_loc=bounds,
        activation=int(activated),
        input_layout="NTD" if bounds is not None else "BNSD",
    )
    return dx.unsqueeze(0) if bounds is not None else dx, dw.t(), db if has_bias else None
