# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)

from fairseq.models.roberta.model import RobertaModel, base_architecture


@register_model("multi_task_roberta")
class MultiTaskRobertaModel(RobertaModel):
    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ):
        assert classification_head_name is not None

        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        feature_x = extra["inner_states"][-1].transpose(0, 1)

        if classification_head_name is not None:
            extra["classification_out"] = self.classification_heads[
                classification_head_name
            ](feature_x)
        return x, extra


@register_model_architecture("multi_task_roberta", "multi_task_roberta_small")
def roberta_small_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    base_architecture(args)
