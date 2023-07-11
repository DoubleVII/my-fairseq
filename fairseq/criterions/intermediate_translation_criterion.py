import torch

# from fairseq import metrics, utils
from fairseq.criterions import register_criterion
import torch.nn.functional as F

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

import sacrebleu


def weighted_label_smoothed_nll_loss(
    lprobs, target, epsilon, weights, ignore_index=None, reduce=True
):
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
        nll_loss = nll_loss * weights.unsqueeze(1) / torch.mean(weights)
        smooth_loss = smooth_loss * weights.unsqueeze(1) / torch.mean(weights)
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("intermediate_translation_criterion")
class IntermediateCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self, task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy,
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=ignore_prefix_size,
            report_accuracy=report_accuracy,
        )

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target, weights = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = weighted_label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            weights,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)

        pred_batch = torch.max(lprobs, dim=-1, keepdim=False)[1]

        def get_sentence_bleu(pred, target):
            pred = pred.tolist()
            target = target.tolist()
            end_position = len(target)
            for i in range(len(target), 0, -1):
                if target[i - 1] != 1:
                    end_position = i
                    break
            pred = pred[:end_position]
            target = target[:end_position]
            pred_str = " ".join(map(str, pred))
            target_str = " ".join(map(str, target))
            bleu_score = sacrebleu.sentence_bleu(
                hypothesis=pred_str, references=[target_str]
            ).score
            if bleu_score < 1e-2:
                bleu_score = 1e-10
            return bleu_score

        blue_scores = torch.FloatTensor(
            list(map(get_sentence_bleu, pred_batch, target))
        ).to(lprobs)
        # print(blue_scores)
        # print(blue_scores.size())

        ids = sample["id"]
        for id_item in set(ids.tolist()):
            blue_scores[ids == id_item] = F.softmax(blue_scores[ids == id_item], dim=0)

        # print(blue_scores)

        weights = blue_scores.unsqueeze(1) * torch.ones((1, target.size(-1))).to(
            blue_scores
        )
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1), weights.view(-1)

