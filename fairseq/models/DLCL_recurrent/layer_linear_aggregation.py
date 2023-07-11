# import math
from typing import Dict, Optional, Tuple

import torch

# import torch.nn.functional as F
from torch import Tensor, nn

# from torch.nn import Parameter

from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout

# from fairseq.modules.quant_noise import quant_noise


"""
    This block just aggregate the encoder layer outputs by simple linear method,
    e.g. a1*layer1 + a2*layer2 + ... + an*layern
"""


class LayerLinearAggregation(nn.Module):
    def __init__(
        self, layer_num, dropout=0.0,
    ):
        super().__init__()

        self.layer_num = layer_num

        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)

        self.layer_weight = nn.Parameter(
            torch.zeros((self.layer_num), requires_grad=True), requires_grad=True,
        )

    def forward(self, x) -> Tuple[Tensor, Optional[Tensor]]:
        """
            x: shape(L, T, B, C)
        """
        norm_layer_weight_float = utils.softmax(self.layer_weight, dim=-1)
        norm_layer_weight = norm_layer_weight_float.type_as(self.layer_weight)
        x = torch.matmul(x.permute(1, 2, 3, 0), norm_layer_weight)
        x = self.dropout_module(x)
        return x
