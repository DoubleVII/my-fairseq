import torch
import torch.nn as nn
from fairseq import utils

LAYER_COMBINE_TYPE = ["base", "linear", "time-wise", "time-wise-ind"]


def build_combinator(type, layer_num, embed_dim):
    assert type in LAYER_COMBINE_TYPE

    if type == "base":
        return BaseCombinator()
    elif type == "linear":
        return LinearCombinator(layer_num)
    elif type == "time-wise-ind":
        return PositionWiseLinearCombinator(layer_num, embed_dim)
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
        return layers[-1]


class LinearCombinator(BaseCombinator):
    def __init__(self, layer_num):
        super().__init__()
        self.layer_num = layer_num
        weights_list = [
            nn.Parameter(torch.zeros((i + 1)))
            for i in range(self.layer_num + 1)  # plus 1 to fit embedding layer
        ]
        self.combine_weights = nn.ParameterList(weights_list)

    def forward(self, layers):
        current_layer_num = len(layers) - 1
        combine_weight = self.combine_weights[current_layer_num]

        norm_combine_weight_float = utils.softmax(combine_weight, dim=-1)
        norm_combine_weight = norm_combine_weight_float.type_as(combine_weight)

        layers_tensor = torch.stack(layers, dim=-1)
        return torch.matmul(layers_tensor, norm_combine_weight)


class PositionWiseLinearCombinator(BaseCombinator):
    def __init__(self, layer_num, embed_dim):
        super().__init__()
        self.layer_num = layer_num
        weights_list = [
            nn.Parameter(torch.zeros((i + 1, embed_dim)))
            for i in range(1, self.layer_num + 1)
        ]
        self.combine_weights = nn.ParameterList(weights_list)

    def forward(self, layers):
        current_layer_num = len(layers) - 1
        if current_layer_num == 0:
            return layers[0]

        (len_size, batch_size, channel) = layers[-1].size()
        input_features = torch.stack(layers, dim=-2).view(
            len_size * batch_size, -1, channel
        )  # T*B x L x C

        output_features = (
            torch.matmul(
                input_features.unsqueeze(-2),
                self.combine_weights[current_layer_num - 1].unsqueeze(-1),
            )
            .squeeze(-1)
            .squeeze(-1)
            .view(len_size, batch_size, -1)
        )  # T x B x L

        norm_combine_weight_float = utils.softmax(output_features, dim=-1)
        norm_combine_weight = norm_combine_weight_float.type_as(output_features)

        layers_tensor = torch.stack(layers, dim=-1)  # T x B x C x L
        return torch.matmul(layers_tensor, norm_combine_weight.unsqueeze(-1)).squeeze(-1)

