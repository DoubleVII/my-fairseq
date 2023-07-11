import torch
import torch.nn as nn
from fairseq import utils

LAYER_COMBINE_TYPE = ["low", "high", "all", "time-wise", "time-wise-ind"]


def build_combinator(type, layer_recurrent, max_layer_num):
    assert type in LAYER_COMBINE_TYPE
    if type == "low":
        return ShallowCombinator(layer_recurrent)
    elif type == "high":
        return DeepCombinator()
    elif type == "all":
        return LinearCombinator(max_layer_num)
    else:
        raise NotImplementedError


class BaseCombinator(nn.Module):
    def __init__(self):
        super(BaseCombinator, self).__init__()
        # self.normalize_before = (
        #     args.encoder_normalize_before if is_encoder else args.decoder_normalize_before
        # )

        # the first layer (aka. embedding layer) does not have layer normalization
        # layers = args.encoder_layers if is_encoder else args.decoder_layers
        # if not hasattr(args, "k"):
        #     args.k = (args.encoder_layers + 1) * [0]
        # layers = len(args.k) - 1 if is_encoder else args.decoder_layers
        # dim = args.encoder_embed_dim if is_encoder else args.decoder_embed_dim
        # self.layer_norms = nn.ModuleList(LayerNorm(dim) for _ in range(layers))

    def forward(self, layers):
        raise NotImplementedError


class ShallowCombinator(BaseCombinator):
    def __init__(self, layer_recurrent):
        super().__init__()
        self.layer_recurrent = layer_recurrent

    def forward(self, layers):
        return layers[-self.layer_recurrent]


class DeepCombinator(BaseCombinator):
    def __init__(self):
        super().__init__()

    def forward(self, layers):
        return layers[-1]


class LinearCombinator(BaseCombinator):
    def __init__(self, max_layer_num):
        super().__init__()
        self.max_layer_num = max_layer_num
        weights_list = [
            nn.Parameter(torch.zeros((i + 1))) for i in range(self.max_layer_num)
        ]
        self.combine_weights = nn.ParameterList(weights_list)

    def forward(self, layers):
        layer_num = len(layers)
        combine_weight = self.combine_weights[layer_num - 1]

        norm_combine_weight_float = utils.softmax(combine_weight, dim=-1)
        norm_combine_weight = norm_combine_weight_float.type_as(combine_weight)
        layers_tensor = torch.stack(layers, dim=-1)
        return torch.matmul(layers_tensor, norm_combine_weight)
