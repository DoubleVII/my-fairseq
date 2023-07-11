import torch

# from fairseq import metrics, utils
from fairseq.criterions import register_criterion
import torch.nn.functional as F

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


def weighted_label_smoothed_nll_loss(
    lprobs, target, epsilon, ids, ignore_index=None, reduce=True
):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)  # batch x seq_len x 1
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)

    nll_loss = nll_loss.squeeze(dim=-1)  # batch x seq_len
    smooth_loss = smooth_loss.squeeze(dim=-1)
    sum_loss = nll_loss.sum(dim=-1)

    for id_item in set(ids.tolist()):
        sum_loss_by_id = sum_loss[ids == id_item]
        mean_loss_by_id = torch.mean(sum_loss_by_id)
        nll_loss[(ids == id_item) & (sum_loss < mean_loss_by_id), :] = 0
        smooth_loss[(ids == id_item) & (sum_loss < mean_loss_by_id), :] = 0
    nll_loss = nll_loss.view(-1)
    smooth_loss = smooth_loss.view(-1)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("intermediate_translation_hard_criterion")
class IntermediateHardCriterion(LabelSmoothedCrossEntropyCriterion):
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
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        ids = sample["id"]
        loss, nll_loss = weighted_label_smoothed_nll_loss(
            lprobs, target, self.eps, ids, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

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
        return lprobs, target

