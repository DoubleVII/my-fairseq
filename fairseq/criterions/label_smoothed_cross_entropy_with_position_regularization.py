import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

import random
import math


@register_criterion("label_smoothed_cross_entropy_with_position_regularization")
class LabelSmoothedCrossEntropyWithPositionRegularizationCriterion(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        position_regularization_lambda,
        max_regularization_position,
        max_regularization_offset,
        regularization_encoder_attention,
        regularization_first_layer,
        attn_heads=8,
    ):
        super().__init__(task, sentence_avg, label_smoothing)
        self.position_regularization_lambda = position_regularization_lambda
        self.max_regularization_position = max_regularization_position
        self.max_regularization_offset = max_regularization_offset
        self.include_encoder_attn = regularization_encoder_attention
        self.regularization_first_layer = regularization_first_layer
        self.attn_heads = attn_heads
        self.MSE_loss = torch.nn.MSELoss(reduction="sum")
        # print("---------------------------------------------")
        # print(self.position_regularization_lambda)
        # print(self.max_regularization_position)
        # print(self.max_regularization_offset)
        # print(self.include_encoder_attn)
        # print(self.attn_heads)
        # print(self.ignore_prefix_size)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument(
            "--position-regularization-lambda",
            default=0.05,
            type=float,
            metavar="D",
            help="weight for the postion regularization loss",
        )
        parser.add_argument(
            "--max-regularization-position",
            default=100,
            type=int,
            metavar="D",
            help="max regularization postion without offset",
        )
        parser.add_argument(
            "--max-regularization-offset",
            default=50,
            type=int,
            metavar="D",
            help="max regularization offset of postion",
        )
        parser.add_argument(
            "--regularization-encoder-attention",
            default=False,
            action="store_true",
            help="also perform postion regularization in encoder-attention",
        )
        parser.add_argument(
            "--regularization-first-layer",
            default=False,
            action="store_true",
            help="only perform postion regularization in first layer",
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        position_regularization_loss = self.compute_position_regularization_loss(
            model, sample["target"].device
        )

        # print(loss)
        # print(self.position_regularization_lambda)

        n_pos_tokens = self.max_regularization_position * self.max_regularization_position
        loss += self.position_regularization_lambda * position_regularization_loss
        # print(loss)

        sentence_size = sample["target"].size(0)

        sample_size = sentence_size if self.sentence_avg else sample["ntokens"]
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "pos_loss": utils.item(position_regularization_loss.data)
            if reduce
            else position_regularization_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "n_pos_tokens": n_pos_tokens,
        }

        return loss, sample_size, logging_output

    def compute_position_regularization_loss(self, model, device):
        # calculate the postion regularization loss
        position_index = torch.ones(
            self.max_regularization_position, dtype=torch.long
        ).to(device=device)
        position_index = torch.cumsum(position_index, dim=0).long()
        position_embedding = model.encoder.embed_positions.weights.index_select(
            0, position_index
        ).detach()  # sen_len x c

        position_offset = random.randint(0, self.max_regularization_offset)
        offset_position_index = torch.ones(
            self.max_regularization_position, dtype=torch.long
        ).to(device=device)
        offset_position_index = (
            torch.cumsum(offset_position_index, dim=0).long() + position_offset
        )
        offset_position_embedding = model.encoder.embed_positions.weights.index_select(
            0, offset_position_index
        ).detach()  # sen_len x dim

        qeury_weights = []
        qeury_bias = []
        key_weights = []
        key_bias = []
        for layer in model.encoder.layers:
            qeury_weights.append(layer.self_attn.q_proj.weight)  # out x in
            qeury_bias.append(layer.self_attn.q_proj.bias)
            key_weights.append(layer.self_attn.k_proj.weight)
            key_bias.append(layer.self_attn.k_proj.bias)
            if self.regularization_first_layer:
                break

        for layer in model.decoder.layers:
            qeury_weights.append(layer.self_attn.q_proj.weight)  # out x in
            qeury_bias.append(layer.self_attn.q_proj.bias)
            key_weights.append(layer.self_attn.k_proj.weight)
            key_bias.append(layer.self_attn.k_proj.bias)
            if self.include_encoder_attn:
                qeury_weights.append(layer.encoder_attn.q_proj.weight)  # out x in
                qeury_bias.append(layer.encoder_attn.q_proj.bias)
                key_weights.append(layer.encoder_attn.k_proj.weight)
                key_bias.append(layer.encoder_attn.k_proj.bias)
            if self.regularization_first_layer:
                break

        attn_num = len(qeury_weights)
        qeury_weights = torch.stack(qeury_weights, dim=0)  # attn_num x out_c x in_c
        qeury_bias = torch.stack(qeury_bias, dim=0)  # attn_num x out_c
        key_weights = torch.stack(key_weights, dim=0)  # attn_num x out_c x in_c
        key_bias = torch.stack(key_bias, dim=0)  # attn_num x out_c

        position_embedding = position_embedding.expand(
            attn_num, -1, -1
        )  # attn_num x sen_len x c

        offset_position_embedding = offset_position_embedding.expand(attn_num, -1, -1)

        q = torch.bmm(
            qeury_weights, position_embedding.transpose(1, 2)
        ) + qeury_bias.unsqueeze(
            -1
        )  # attn_num x out_c x sen_len
        k = torch.bmm(
            key_weights, position_embedding.transpose(1, 2)
        ) + key_bias.unsqueeze(
            -1
        )  # attn_num x out_c x sen_len

        """
         the two parts of q,k computation is similar and can be fused to accelerate in future
        """
        q_offset = torch.bmm(
            qeury_weights, offset_position_embedding.transpose(1, 2)
        ) + qeury_bias.unsqueeze(
            -1
        )  # attn_num x out_c x sen_len
        k_offset = torch.bmm(
            key_weights, offset_position_embedding.transpose(1, 2)
        ) + key_bias.unsqueeze(
            -1
        )  # attn_num x out_c x sen_len

        q = (
            q.permute(2, 0, 1)
            .contiguous()
            .view(self.max_regularization_position, attn_num * self.attn_heads, -1)
            .transpose(0, 1)
        )  # attn_num * attn_heads x sen_len x out_c
        k = (
            k.permute(2, 0, 1)
            .contiguous()
            .view(self.max_regularization_position, attn_num * self.attn_heads, -1)
            .transpose(0, 1)
        )  # attn_num * attn_heads x sen_len x out_c
        q_offset = (
            q_offset.permute(2, 0, 1)
            .contiguous()
            .view(self.max_regularization_position, attn_num * self.attn_heads, -1)
            .transpose(0, 1)
        )  # attn_num * attn_heads x sen_len x out_c
        k_offset = (
            k_offset.permute(2, 0, 1)
            .contiguous()
            .view(self.max_regularization_position, attn_num * self.attn_heads, -1)
            .transpose(0, 1)
        )  # attn_num * attn_heads x sen_len x out_c

        attn_weights = torch.bmm(
            q, k.transpose(1, 2)
        )  # attn_num * attn_heads x sen_len x sen_len

        attn_weights_offset = torch.bmm(
            q_offset, k_offset.transpose(1, 2)
        )  # attn_num * attn_heads x sen_len x sen_len

        loss = self.MSE_loss(attn_weights_offset, attn_weights)
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))
        pos_loss_sum = utils.item(sum(log.get("pos_loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        n_pos_tokens = utils.item(
            sum(log.get("n_pos_tokens", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "pos_loss", pos_loss_sum / n_pos_tokens / math.log(2), n_pos_tokens, round=3,
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
