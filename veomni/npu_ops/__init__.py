import logging

import torch

# from veomni.models.transformers.flux.modeling_flux import print_rank_0
# from .flash_attn.flash_attn import apply_transformers_attention_patch
from .fully_shard.fully_shard import apply_fully_shard_patch


logger = logging.getLogger(__name__)


def apply_ops_patch():
    apply_transformers_attention_patch()
    # print_rank_0(logger.info, "✅ MindSpeed-MM ops patch applied.")

    # apply modify fully_shard patch
    apply_fully_shard_patch()
    # print_rank_0(logger.info, "✅ MindSpeed-MM fully_shard patch applied.")