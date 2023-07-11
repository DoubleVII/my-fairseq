# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from ntpath import join
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import random


from torch._C import dtype

import torch.nn.functional as F


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    detector_loss_epoch: int = field(
        default=0,
        metadata={"help": "epoch to begin train detector"},
    )
    detector_loss_lambda: float = field(
        default=0.5,
        metadata={"help": "weight for detector loss"},
    )
    positive_loss_weight: float = field(
        default=1.0,
        metadata={"help": "weight for positive class loss"},
    )
    negative_loss_weight: float = field(
        default=1.0,
        metadata={"help": "weight for negative class loss"},
    )
    no_compute_translation_loss: bool = field(
        default=False,
        metadata={"help": "no_compute_translation_loss"},
    )



def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("label_smoothed_cross_entropy_with_noise_detection", dataclass=LabelSmoothedCrossEntropyCriterionConfig)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        detector_loss_epoch,
        detector_loss_lambda,
        positive_loss_weight,
        negative_loss_weight,
        ignore_prefix_size=0,
        no_compute_translation_loss=False,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.detector_loss_epoch = detector_loss_epoch
        self.detector_loss_lambda = detector_loss_lambda
        self.detector_weight = torch.FloatTensor([negative_loss_weight, positive_loss_weight])
        self.no_compute_translation_loss = no_compute_translation_loss

        if self.no_compute_translation_loss:
            assert self.forward_with_detector
            assert detector_loss_epoch == 0


    def forward_with_detector(self, model, sample, reduce=True):
        src_tokens = sample["net_input"]["src_tokens"]

        batch_size = src_tokens.size(0)

        src_lengths = sample["net_input"]["src_lengths"]
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        
        # import pdb
        # pdb.set_trace()
        
        shuffle_index = list(range(batch_size))
        random.shuffle(shuffle_index)
        for i in range(batch_size):
            if shuffle_index[i] == i:
                shuffle_index[i] = (i+1) % batch_size
        append_src_tokens = torch.cat((src_tokens, src_tokens), dim=0)
        append_src_lengths = torch.cat((src_lengths, src_lengths), dim=0)
        append_prev_output_tokens = prev_output_tokens[shuffle_index]
        append_prev_output_tokens = torch.cat((prev_output_tokens, append_prev_output_tokens), dim=0)
        sample["net_input"]["src_tokens"] = append_src_tokens
        sample["net_input"]["src_lengths"] = append_src_lengths
        sample["net_input"]["prev_output_tokens"] = append_prev_output_tokens
        sample["net_input"]["translation_batch_size"] = batch_size
        sample["net_input"]["forward_detector"] = True
        # print("criterions size:", append_prev_output_tokens.size())
        # print("target size:", sample["target"].size())

        net_output = model(**sample["net_input"])
        # print("net output size:", net_output[0].size())
        if not self.no_compute_translation_loss:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        
        detector_target = torch.zeros((batch_size*2,),dtype=torch.long).to(src_tokens)
        detector_target[batch_size:] = 1
        detector_loss, n_correct, total, pred = self.compute_detector_loss(net_output, detector_target, reduce=reduce)
        if not self.no_compute_translation_loss:
            loss = loss +  detector_loss * self.detector_loss_lambda
        else:
            loss = detector_loss * self.detector_loss_lambda
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "detector_loss": detector_loss,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "detection_sentence_num": detector_target.size(0),
        }
        if not self.no_compute_translation_loss:
            logging_output["nll_loss"] = nll_loss.data
        if self.report_accuracy:
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = total
        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.task.epoch > self.detector_loss_epoch:
            return self.forward_with_detector(model, sample, reduce)

        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            # n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = 0
            logging_output["total"] = 1
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss
    
    def compute_detector_loss(self, net_output, target, reduce=True):
        logits = net_output[1]["detector_out"]
        lprobs = utils.log_softmax(logits, dim=-1)

        nll_loss = F.nll_loss(lprobs, target, weight=self.detector_weight.to(lprobs), reduction="sum" if reduce else "none")
        
        # if target.dim() == lprobs.dim() - 1:
        #     target = target.unsqueeze(-1)
        # nll_loss = -lprobs.gather(dim=-1, index=target)
        # nll_loss = nll_loss.squeeze(-1)
        # if reduce:
        #     nll_loss = nll_loss.sum()
        loss = nll_loss

        pred = lprobs.argmax(1)
        bool_pred = pred.to(torch.bool)
        if torch.all(bool_pred) or not torch.any(bool_pred):
            print("over pred detected!")
            # print(pred)
        n_correct = torch.sum(pred.eq(target.squeeze()))
        # print("correct:", n_correct)
        total = target.size(0)
        return loss, n_correct, total, pred


    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        detector_loss_sum = sum(log.get("detector_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        detection_sentence_num = sum(log.get("detection_sentence_num", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        if detection_sentence_num > 0:
            metrics.log_scalar(
                "detector_loss", detector_loss_sum / detection_sentence_num / math.log(2), detection_sentence_num, round=3
            )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
