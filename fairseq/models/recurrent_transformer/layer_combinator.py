import torch
import torch.nn as nn
from fairseq import utils

LAYER_COMBINE_TYPE = ["low", "high", "all", "linear", "time-wise", "time-wise-ind"]


def build_combinator(type, layer_recurrent, layer_num, max_layer_num, embed_dim):
    assert type in LAYER_COMBINE_TYPE
    if type == "low":
        return ShallowCombinator(layer_recurrent)
    elif type == "high":
        return DeepCombinator()
    elif type == "all":
        return LinearCombinator(layer_num, layer_recurrent)
    elif type == "linear":
        return RecurrentLinearCombinator(layer_num, layer_recurrent)
    elif type == "time-wise-ind":
        return PositionWiseRecurrentLinearCombinator(
            layer_num, layer_recurrent, embed_dim
        )
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
        if len(layers) == 1:
            return layers[0]
        return layers[-self.layer_recurrent]


class DeepCombinator(BaseCombinator):
    def __init__(self):
        super().__init__()

    def forward(self, layers):
        return layers[-1]


class LinearCombinator(BaseCombinator):
    def __init__(self, layer_num, layer_recurrent):
        super().__init__()
        self.layer_num = layer_num
        self.layer_recurrent = layer_recurrent
        weights_list = [
            nn.Parameter(torch.zeros((i + 1)))
            for i in range(self.layer_num + 1)  # plus 1 to fit embedding layer
        ]
        self.combine_weights = nn.ParameterList(weights_list)

    def forward(self, layers):
        current_layer_num = len(layers) // self.layer_recurrent
        combine_weight = self.combine_weights[current_layer_num]

        norm_combine_weight_float = utils.softmax(combine_weight, dim=-1)
        norm_combine_weight = norm_combine_weight_float.type_as(combine_weight)

        layers = layers[0 :: self.layer_recurrent]
        layers_tensor = torch.stack(layers, dim=-1)
        return torch.matmul(layers_tensor, norm_combine_weight)


class RecurrentLinearCombinator(BaseCombinator):
    def __init__(self, layer_num, layer_recurrent):
        super().__init__()
        self.layer_num = layer_num
        self.layer_recurrent = layer_recurrent
        weights_list = [
            nn.Parameter(torch.zeros(layer_recurrent)) for _ in range(self.layer_num)
        ]
        self.combine_weights = nn.ParameterList(weights_list)

    def forward(self, layers):
        current_layer_num = len(layers) // self.layer_recurrent
        if current_layer_num == 0:
            return layers[0]
        combine_weight = self.combine_weights[current_layer_num - 1]

        norm_combine_weight_float = utils.softmax(combine_weight, dim=-1)
        norm_combine_weight = norm_combine_weight_float.type_as(combine_weight)
        layers_tensor = torch.stack(layers[-self.layer_recurrent :], dim=-1)
        return torch.matmul(layers_tensor, norm_combine_weight)


class PositionWiseRecurrentLinearCombinator(BaseCombinator):
    def __init__(self, layer_num, layer_recurrent, embed_dim):
        super().__init__()
        self.layer_num = layer_num
        self.layer_recurrent = layer_recurrent
        weights_list = [
            nn.Parameter(torch.zeros((self.layer_recurrent, embed_dim)))
            for _ in range(self.layer_num)
        ]
        self.combine_weights = nn.ParameterList(weights_list)

    def forward(self, layers):
        current_layer_num = len(layers) // self.layer_recurrent
        if current_layer_num == 0:
            return layers[0]

        valid_layers = layers[-self.layer_recurrent :]

        (len_size, batch_size, channel) = valid_layers[-1].size()
        input_features = torch.stack(valid_layers, dim=-2).view(
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
        valid_layers_tensor = torch.stack(valid_layers, dim=-1)  # T x B x C x L
        return torch.matmul(
            valid_layers_tensor, norm_combine_weight.unsqueeze(-1)
        ).squeeze(-1)
