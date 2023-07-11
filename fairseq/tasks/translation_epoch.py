# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks.translation import TranslationConfig
import torch

EVAL_BLEU_ORDER = 4
@register_task("translation_epoch", dataclass=TranslationConfig)
class TranslationEqochTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg,src_dict, tgt_dict)
        self.epoch = 0

    def begin_epoch(self, epoch, model):
        self.epoch = epoch

    def base_valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
    
    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = self.base_valid_step(sample, model, criterion)
        if self.cfg.eval_bleu:
            if self.epoch > criterion.detector_loss_epoch:
                sample["net_input"].pop("forward_detector")
                sample["net_input"].pop("translation_batch_size")
                batch_size = sample["net_input"]["src_tokens"].size(0)
                sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"][:batch_size//2, :]
                sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"][:batch_size//2]
                sample["net_input"]["prev_output_tokens"] = sample["net_input"]["prev_output_tokens"][:batch_size//2, :]
            # import pdb
            # pdb.set_trace()
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output
