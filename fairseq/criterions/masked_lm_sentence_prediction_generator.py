# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("masked_lm_sentence_prediction_generator")
class MaskedLmLossWithSentenceGeneratorPrediction(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        parser.add_argument('--classification-loss-weight',
                            default=1.0,
                            type=float,
                            help='loss weight of the classification task')

    def __init__(
        self, task, classification_head_name, classification_loss_weight, tpu=False
    ):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.classification_loss_weight = classification_loss_weight
        self.tpu = tpu

    def forward(self, model, sample, reduce=True, return_score=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"


        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).


        masked_tokens = None

        logits0, logits1 = model(
            **sample["net_input"],
            features_only=False ,
            return_all_hiddens=True,
            classification_head_name=self.classification_head_name, 
            masked_tokens=masked_tokens,
        )
        
        logits1 = logits1["classification_out"]



        targets1 = sample["label"].view(-1)
        # print(targets1)
        sample_size1 = targets1.numel()
        lprobs = F.log_softmax(logits1, dim=-1, dtype=torch.float32)
        loss1 = F.nll_loss(lprobs, targets1, reduction="sum")

        loss =  loss1 * self.classification_loss_weight

        # print("loss0 here:",loss0)
        # print("loss1 here:",loss1)

        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size1": sample_size1,
            "loss1": loss1,
        }

        preds = logits1.argmax(dim=1)
        logging_output["ncorrect"] = (preds == targets1).sum()
        logging_output["negtive_sample_size"] = targets1.sum()
        logging_output["positive_sample_size"] = len(targets1) - logging_output["negtive_sample_size"]
        logging_output["positive_ncorrect"] = (preds == targets1).to(targets1).masked_fill(targets1 == 1, 0).sum()
        logging_output["negtive_ncorrect"] = (preds == targets1).to(targets1).masked_fill(targets1 == 0, 0).sum()
        
        if return_score:
            return loss, sample_size1, logging_output, lprobs
        return loss, sample_size1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss1_sum = sum(log.get("loss1", 0) for log in logging_outputs)
        sample_size1 = sum(log.get("sample_size1", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss1", loss1_sum / sample_size1 / math.log(2), sample_size1, round=3
        )
        metrics.log_scalar(
            "loss", loss1_sum / sample_size1 / math.log(2), sample_size1, round=3
        )
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            positive_ncorrect = sum(log.get("positive_ncorrect", 0) for log in logging_outputs)
            negtive_ncorrect = sum(log.get("negtive_ncorrect", 0) for log in logging_outputs)
            positive_sample_size = sum(log.get("positive_sample_size", 0) for log in logging_outputs)
            negtive_sample_size = sum(log.get("negtive_sample_size", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )
            metrics.log_scalar(
                "positive_accuracy", 100.0 * positive_ncorrect / positive_sample_size, positive_sample_size, round=1
            )
            metrics.log_scalar(
                "negtive_accuracy", 100.0 * negtive_ncorrect / negtive_sample_size, negtive_sample_size, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
