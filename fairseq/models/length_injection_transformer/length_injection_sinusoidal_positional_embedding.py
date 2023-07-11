# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import torch
from fairseq import utils
from torch import Tensor, nn
import math


class LengthInjectionSinusoidalPositionalEmbedding(nn.Module):
    """This module produces adaptive frequency sinusoidal positional embeddings.

    Padding symbols are ignored.
    """

    def __init__(
        self, embedding_dim, padding_idx, phase=False, freq_base=200.0, static_param=False
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.freq_base = freq_base
        self.log_freq_base = math.log(freq_base)

        # half_dim = self.embedding_dim // 2
        assert self.embedding_dim % 4 == 0
        quarter_dim = self.embedding_dim // 4
        if static_param:
            self.forward_freq_weights = torch.linspace(
                0, 1, quarter_dim, dtype=torch.float, requires_grad=False
            )
            self.backward_freq_weights = torch.linspace(
                0, 1, quarter_dim, dtype=torch.float, requires_grad=False
            )
        else:
            self.forward_freq_weights = nn.Parameter(
                torch.linspace(0, 1, quarter_dim, dtype=torch.float)
            )
            self.backward_freq_weights = nn.Parameter(
                torch.linspace(0, 1, quarter_dim, dtype=torch.float)
            )

        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen x 2] or [seqlen x bsz x 2], the elements of input must be
        the position."""
        # print(input)
        # assert False

        bsz, seqlen, _ = input.size()

        freq_weights = torch.stack(
            (self.forward_freq_weights, self.backward_freq_weights), dim=0
        ).to(self._float_tensor)

        forward_freqs = torch.exp(freq_weights * -self.log_freq_base)  # 2 x quarter_dim
        out_phase = input.unsqueeze(-1) * forward_freqs.unsqueeze(0).unsqueeze(
            0
        )  # bsz x seqlen x 2 x quarter_dim
        out_phase = out_phase.view(bsz, seqlen, -1)

        output = torch.cat(
            [torch.sin(out_phase), torch.cos(out_phase)], dim=-1
        )  # bsz x seqlen x dim

        mask = input[:, :, 0].eq(self.padding_idx).unsqueeze(-1)  # bsz x seqlen x 1
        output = output.masked_fill(mask, 0.0)

        return output

    # @staticmethod
    # def get_embedding(
    #     num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    # ):
    #     """Build sinusoidal embeddings.

    #     This matches the implementation in tensor2tensor, but differs slightly
    #     from the description in Section 3.5 of "Attention Is All You Need".
    #     """
    #     half_dim = embedding_dim // 2
    #     emb = math.log(10000) / (half_dim - 1)
    #     emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    #     emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
    #         1
    #     ) * emb.unsqueeze(0)
    #     emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    #     if embedding_dim % 2 == 1:
    #         # zero pad
    #         emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    #     if padding_idx is not None:
    #         emb[padding_idx, :] = 0
    #     return emb
