from fairseq.criterions import FairseqCriterion, register_criterion
from torch import nn
from typing import List, Dict, Any
from fairseq import metrics
import math


@register_criterion("teach_recurrent_criterion")
class TeachRecCriterion(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)
        self.criterion = nn.MSELoss(reduction="none")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        pass

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample["net_input"])
        loss = self.compute_loss(net_output)
        sample_size = sample["ntokens"]
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, net_output):
        teacher_encoder_out = net_output.teacher_encoder_out
        recurrent_encoder_out = net_output.recurrent_encoder_out
        encoder_padding_mask = teacher_encoder_out.encoder_padding_mask  # B x T
        allloses = self.criterion(
            recurrent_encoder_out.encoder_out, teacher_encoder_out.encoder_out
        )  # T x B x C
        loss = (
            allloses.masked_fill(
                encoder_padding_mask.transpose(0, 1).unsqueeze(-1), float(0)
            )
            .mean(dim=-1)
            .sum()
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
