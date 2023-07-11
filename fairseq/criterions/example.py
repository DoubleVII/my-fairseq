from fairseq.criterions import FairseqCriterion, register_criterion
from torch import nn
from typing import List, Dict, Any
from fairseq import metrics


@register_criterion("example_criterion")
class ExampleCriterion(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)
        self.criterion = nn.MSELoss(reduction="mean")

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
        output = model(sample["nums"])
        # print("input:", sample["nums"])
        # print("output", output)
        # print("target", sample["targets"])
        loss = self.criterion(output, sample["targets"])
        sample_size = len(sample["nums"])
        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
