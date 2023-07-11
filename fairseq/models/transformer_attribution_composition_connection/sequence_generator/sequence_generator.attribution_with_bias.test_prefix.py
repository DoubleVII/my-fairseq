# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

import torch
from torch import LongTensor
import torch.nn as nn
from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)


class SequenceGenerator(nn.Module):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
    ):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert temperature > 0, "--temperature must be greater than 0"

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
            hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
        )

        self.model.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

        self.bias_ratio = []
        self.ratio_count = []
        for i in range(200):
            self.bias_ratio.append(torch.zeros((1,)).cuda())
            self.ratio_count.append(0)
        self.count = 0

        # cast model to double
        self.model.double()

        self.src_dict = self.model.models[0].encoder.dictionary

        self.drop_probs = []

        self.analysis_info_list = []
        self.data_rec = []
        logger.info("begin inference")

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    # TODO(myleott): unused, deprecate after pytorch-translate migration
    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if "net_input" not in s:
                continue
            input = s["net_input"]
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {k: v for k, v in input.items() if k != "prev_output_tokens"}
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]["tokens"]) for h in hypos))
            for i, id in enumerate(s["id"].data):
                # remove padding
                src = utils.strip_pad(input["src_tokens"].data[i, :], self.pad)
                ref = (
                    utils.strip_pad(s["target"].data[i, :], self.pad)
                    if s["target"] is not None
                    else None
                )
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):

        # self.analysis_info_list.append([])
        analysis_info_list = []

        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]
        # self.hack_input(net_input)
        prefix_tokens = self.hack_input(net_input)

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input")

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        return_connection = False
        net_input["return_compositions"] = True
        net_input["return_connection"] = return_connection
        encoder_outs = self.model.forward_encoder(net_input)

        encoder_compositions_states = None
        encoder_connection_states = None
        encoder_attn_states = None
        if "compositions_states" in encoder_outs[0]._fields:
            encoder_compositions_states = encoder_outs[0].compositions_states
        if "connection_states" in encoder_outs[0]._fields:
            encoder_connection_states = encoder_outs[0].connection_states
        if "attn_states" in encoder_outs[0]._fields:
            encoder_attn_states = encoder_outs[0].attn_states
        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        decomposition_scores = torch.zeros(bsz * beam_size, max_len + 1, 2 * max_len + 1)

        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(encoder_outs, reorder_state)

            (
                lprobs,
                avg_attn_scores,
                compositions_logits,
                analysis_info,
            ) = self.model.forward_decoder(
                tokens[:, : step + 1],
                encoder_outs,
                incremental_states,
                self.temperature,
                return_compositions=False,
                return_connection=return_connection,
            )
            # analysis_info_list.append(analysis_info)

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            # decomposition_scores = decomposition_scores.type_as(compositions_logits)

            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )
            beam_cand_indices = cand_indices + self.vocab_size * cand_beams

            # decomposition_cand_scores = compositions_logits.view(
            #     bsz, -1, src_len + step + 2
            # ).gather(
            #     dim=1,
            #     index=beam_cand_indices.unsqueeze(-1).expand(-1, -1, src_len + step + 2),
            # )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )
                # decomposition_eos_scores = decomposition_cand_scores[:, :beam_size][
                #     eos_mask[:, :beam_size]
                # ]

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                    None,
                    None,
                    src_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(bsz, dtype=torch.bool, device=cand_indices.device)
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(bsz, device=cand_indices.device).masked_select(
                    batch_mask
                )

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                # decomposition_cand_scores = decomposition_cand_scores[batch_idxs]

                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                # decomposition_scores = decomposition_scores.view(
                #     bsz, -1, decomposition_scores.size(-1)
                # )[batch_idxs].view(new_bsz * beam_size, -1, decomposition_scores.size(-1))

                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
                # decomposition_scores[:, :step] = torch.index_select(
                #     decomposition_scores[:, :step], dim=0, index=active_bbsz_idx
                # )

            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # decomposition_scores[:, :, : src_len + step + 2].view(
            #     bsz, beam_size, -1, src_len + step + 2
            # )[:, :, step] = torch.gather(
            #     decomposition_cand_scores,
            #     dim=1,
            #     index=active_hypos.unsqueeze(-1).expand(-1, -1, src_len + step + 2),
            # )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(List[Dict[str, Tensor]], finalized[sent])
            # pad_num = (src_tokens[sent] == self.tgt_dict.pad()).sum()
            # self.decomposition_positional_scores.append(
            #     finalized[sent][0]["decomposition_positional_scores"][:, pad_num:].cpu()
            # )
            # self.positional_scores.append(finalized[sent][0]["positional_scores"].cpu())

            # input_sum = finalized[sent][0]["decomposition_positional_scores"][
            #     :, 1 + pad_num :
            # ].sum(dim=-1)
            # bias_sum = finalized[sent][0]["decomposition_positional_scores"][:, 0]
            # for i in range(len(decomposition_positional_scores)):
            #     class_len = (decomposition_positional_scores.size(1) - 1 - pad_num) - (
            #         decomposition_positional_scores[i, pad_num + 1 :] == 0.0
            #     ).to(torch.long).sum().item()
            #     self.bias_ratio[class_len] += (
            #         bias_sum[i].abs() / (bias_sum[i] + input_sum[i]).abs()
            #     )
            #     self.ratio_count[class_len] += 1

        # decomposition_positional_scores = finalized[0][0][
        #     "decomposition_positional_scores"
        # ]
        net_input = sample["net_input"]

        # self.compute_scores(
        #     decomposition_positional_scores,
        #     net_input,
        #     finalized[0][0]["tokens"],
        #     finalized[0][0]["positional_scores"],
        #     bos_token,
        #     drop_rate=0.1,
        # )

        self.count += len(sample["id"])

        torch.set_printoptions(precision=2, sci_mode=False, linewidth=150)

        def print_set_logits(words_set: List[str], step: int):
            res = []
            for word in words_set:
                res.append(
                    check_composition_logit(analysis_info_list, step, word).tolist()
                )
            for item in res:
                print(*item)

        def cal_cos_sim(t1: Tensor, t2: Tensor, dim=-1):
            return (t1 * t2).sum(dim=dim) / (t1.norm(dim=dim) * t2.norm(dim=dim))

        def check_composition_pred(
            analysis_info_list, step, composition_id=None, topk=5, layer=-1, largest=True,
        ):
            select_composition_states = analysis_info_list[step]["compositions_states"][
                layer
            ]
            token_embed = self.model.models[0].decoder.output_projection.weight
            token_dict = self.tgt_dict

            if composition_id is None:
                select_composition_states = select_composition_states.sum(dim=1)[0, 0]
            else:
                select_composition_states = select_composition_states[
                    0, composition_id, 0
                ]

            select_composition_logits = select_composition_states @ token_embed.t()

            topk_scores, topk_index = select_composition_logits.topk(
                k=topk, largest=largest
            )
            for score, index in zip(topk_scores, topk_index):
                print(score.item(), token_dict.string(index[None]))

        def check_composition_logit(
            analysis_info_list, step, token_str: str = None, layer=-1
        ):
            select_composition_states = analysis_info_list[step]["compositions_states"][
                layer
            ][0, :, 0]
            token_embed = self.model.models[0].decoder.output_projection.weight
            token_dict = self.tgt_dict
            if token_str == "<eos>":
                token_index = token_dict.eos()
            else:
                token_index = token_dict.index(token_str)
            assert token_index != token_dict.unk()
            select_composition_logits = (
                select_composition_states @ token_embed[token_index][None].t()
            )
            select_composition_logits = select_composition_logits.view(-1)
            print(((select_composition_logits * 100).round() / 100))
            return select_composition_logits

        # check_composition_pred(analysis_info_list, 0)

        # exit()
        import pdb

        pdb.set_trace()
        return finalized

        sample_num = len(self.src_strs)

        self.decoder_norm_logit_analysis(
            analysis_info_list,
            finalized[0][0]["tokens"],
            sample_num=sample_num,
            file_name="connection_test_decoder_norms_logits.pt",
        )

        self.decoder_attn_analysis(
            analysis_info_list,
            sample_num=sample_num,
            file_name="connection_test_decoder_attn.pt",
        )

        self.decoder_connection_analysis(
            analysis_info_list,
            sample_num=sample_num,
            file_name="connection_test_decoder_connection.pt",
        )

        if encoder_compositions_states is not None:
            self.encoder_norm_analysis(
                encoder_compositions_states,
                sample_num=sample_num,
                file_name="connection_test_encoder_norms.pt",
            )
        if encoder_attn_states is not None:
            self.encoder_attn_analysis(
                encoder_attn_states,
                sample_num=sample_num,
                file_name="connection_test_encoder_attn.pt",
            )

        if encoder_connection_states is not None:
            self.encoder_connection_norm_analysis(
                encoder_connection_states,
                sample_num=sample_num,
                file_name="connection_test_encoder_connection_norms.pt",
            )

        # self.tail_peak_record(
        #     src_tokens[0],
        #     finalized[0][0]["tokens"],
        #     decomposition_positional_scores,
        #     analysis_info_list,
        #     sample_num=3003,
        # )

        return finalized

    def save_tensor_infos(
        self, keys, values, sample_num=10, file_name="connection_tensor_infos.pt"
    ):
        if "connection_tensor_infos" not in dir(self):
            self.connection_tensor_infos = []
        info_dict = {}
        for key, value in zip(keys, values):
            info_dict[key] = value
        self.connection_tensor_infos.append(info_dict)

        if self.count == sample_num:
            torch.save(self.connection_tensor_infos, file_name)

    def tail_peak_record(
        self,
        src_tokens,
        tokens,
        decomposition_positional_scores,
        analysis_info_list,
        sample_num=10,
        file_name="tail_peak.pt",
    ):
        if "logit_list" not in dir(self):
            self.logit_list = []
        if "norm_list" not in dir(self):
            self.norm_list = []

        prev_tokens = torch.zeros_like(tokens).to(tokens)
        prev_tokens[0] = self.tgt_dict.bos()
        prev_tokens[1:] = tokens[:-1]

        def remove_bpe_word(tokens, pos):
            def get_word_str(word_id):
                if word_id == self.tgt_dict.eos():
                    return "<eos>"
                elif word_id == self.tgt_dict.bos():
                    return "<bos>"
                elif word_id == self.tgt_dict.unk():
                    return "<unk>"
                elif word_id == self.tgt_dict.pad():
                    return "<pad>"
                else:
                    return self.tgt_dict.string([word_id])

            res_list = []
            # forward search
            for i in range(pos - 1, -1, -1):
                word_str = get_word_str(tokens[i])
                if word_str.endswith("@@"):
                    res_list.insert(0, word_str[:-2])
                else:
                    break
            # backward search
            for i in range(pos, len(tokens)):
                word_str = get_word_str(tokens[i])
                if word_str.endswith("@@"):
                    res_list.append(word_str[:-2])
                else:
                    res_list.append(word_str)
                    break
            return "".join(res_list)

        def extract_data(scores_mat: Tensor):
            pos_range = min(5, scores_mat.size(0))
            topk = 3
            src_len = scores_mat.size(1) - 1 - scores_mat.size(0)

            scores_mat: Tensor = scores_mat[-pos_range:, 1:]
            _, top_index = scores_mat.topk(topk, dim=-1)

            score_pos_list = []

            for i in range(pos_range):
                score_dict = {}
                pos_top_index = top_index[i]
                score_dict["is_tgt"] = pos_top_index >= src_len
                score_dict["top_word"] = []
                for j in range(len(pos_top_index)):
                    if score_dict["is_tgt"][j]:
                        score_dict["top_word"].append(
                            remove_bpe_word(prev_tokens, pos_top_index[j] - src_len)
                        )
                    else:
                        score_dict["top_word"].append(
                            remove_bpe_word(src_tokens, pos_top_index[j])
                        )
                score_pos_list.append(score_dict)
            return score_pos_list

        logit_pos_list = extract_data(decomposition_positional_scores)
        self.logit_list.append(logit_pos_list)

        compositions_states_norms = torch.zeros(
            (
                len(analysis_info_list),
                analysis_info_list[-1]["compositions_states"][0].size(1),
            )
        )
        for j, analysis_info in enumerate(analysis_info_list):
            compositions = analysis_info["compositions_states"][-1]
            compositions_states_norms[j, : compositions.size(1)] = torch.norm(
                compositions, dim=-1
            )[0, :, 0]

        norm_pos_list = extract_data(compositions_states_norms)
        self.norm_list.append(norm_pos_list)

        if self.count == sample_num:
            torch.save((self.logit_list, self.norm_list), file_name)

    def hack_input(self, net_input):
        def load_nh_test_data():
            if "src_strs" not in dir(self):
                self.src_strs = []
            else:
                return
            base_path = "/home/yangs/code/fairseq_models/attribution_transformer_wmt16/hallucination/"
            with open(base_path + "test_normal.txt", "r") as normal_file:
                for line in normal_file:
                    self.src_strs.append(line.strip())
            with open(base_path + "test_nh.txt", "r") as nh_file:
                for line in nh_file:
                    self.src_strs.append(line.strip())
            assert len(self.src_strs) == 100

        # src_strs = [
        #     "The staff of this hotel is very nice .",
        #     "The room of this ship is very nice .",
        #     "The staff of this ship is very nice .",
        #     "The price of this ship is very nice .",
        #     "The window of this ship is very nice .",
        # ]
        # load_nh_test_data()
        # self.src_strs = [
        #     "I hope he sees what I am doing .",
        #     "A lot of my clients are very young .",
        #     "The price of this ship is very nice .",
        #     "The room of this ship is very nice .",
        #     "The room of this hotel is very nice .",
        # ]
        # self.src_strs = [
        #     "Revolu@@ tionary Sac@@ red Music Fac@@ tory",
        #     "Airport building ev@@ acu@@ ated",
        #     "A lot of my clients are very young .",
        # ]

        # self.src_strs = [
        #     "a lot of my clients are very young .",
        #     "a lot of my teachers are very young .",
        #     "a lot of my friends are very young .",
        # ]
        # self.src_strs = [
        #     "A lot of my clients are very young .",
        #     "A lot of my teachers are very young .",
        #     "A lot of my friends are very young .",
        # ]

        # self.src_strs = [
        #     "I have cut the tree .",
        #     "I will cut the tree .",
        # ]

        self.src_strs = [
            "俄 专家 : 太空 梭 灾难 可能 导致 太空@@ 站 暂停 运作",
            "德 电信 产业 mo@@ bil@@ com 创办 人 施@@ 密@@ 德 宣布 破产",
        ]

        # self.src_strs = [
        #     "德 电信 产业 mo@@ bil@@ com 创办 人 密@@ 德@@ 霍夫 宣布 破产",
        #     "俄 专家 : 太空 梭 灾难 可能 导致 太空@@ 站 暂停 运作",
        # ]

        # self.src_strs = [
        #     "( 法新社 莫斯科 电 ) 一 位 俄罗斯 太空 专家 指出 , 美国 太空 梭 哥伦比亚 号 在 返回 地球 途中 解体 的 不幸 事件 可能 会 导致 靠 美国 太空 梭 及 俄罗斯 太空船 进行 的 国际 太空@@ 站 人 为 任务 暂时 停止 。",
        #     "俄罗斯 专家 指出 , 一日 的 不幸 事件 显示 , 太空 梭 的 不 可靠 , 俄罗斯 在 一九七一年 也 曾 出 过 意外 , 当时 联合 十一 号 太空船 在 重返 地球 时 失事 , 折 损 了 太空 船上 的 三 名 太空 人 。"
        # ]

        self.tgt_strs = [
            "russian experts : space v@@",
            "russian experts : space r@@ ous@@ in disaster likely to cause temporary suspension of operation of space station",
            "sch@@ il@@ t , founder of mobile telecom industry , declared bankruptcy",
        ]

        # self.tgt_strs = [
        #     "( afp report from moscow ) a russian space expert pointed out that the unfortunate incident of the us space colombian plane 's collapse on its way back to the earth may cause the international space station , which relies on the us space craft and russia 's spacecraft , to temporarily stop its mission .",
        #     "russian experts pointed out that the one day unfortunate incident showed the im@@ reliability of spacecraft , and russia also had an accident in 1971 . at that time , the united spacecraft no. 11 crashed into the earth when it returned to the earth , damaging three space people on board a space ship ."
        # ]

        # self.tgt_strs = [
        #     "mid@@ del@@ ho@@ ff , founder of german telecom industry , declared bankruptcy",
        #     "russian experts : space r@@ ous@@ sef disaster likely to cause temporary suspension of operation",
        # ]

        if self.count < len(self.src_strs):
            src_string = self.src_strs[self.count]
            if self.count < len(self.tgt_strs):
                tgt_string = self.tgt_strs[self.count]
            else:
                tgt_string = None
        else:
            exit()
        new_src_tokens = self.src_dict.encode_line(src_string, add_if_not_exist=False).to(
            net_input["src_tokens"]
        )

        new_tgt_tokens = None
        if tgt_string is not None:
            new_tgt_tokens = self.tgt_dict.encode_line(
                tgt_string, add_if_not_exist=False
            ).to(net_input["src_tokens"])
            assert not torch.any(new_tgt_tokens == self.tgt_dict.unk())

        assert not torch.any(new_src_tokens == self.src_dict.unk())
        new_src_tokens = new_src_tokens[None]
        net_input["src_tokens"] = new_src_tokens
        net_input["src_lengths"][0] = new_src_tokens.size(-1)
        if new_tgt_tokens is not None:
            return new_tgt_tokens[None, :-1]
        else:
            return None

    def decoder_attn_analysis(
        self, analysis_info_list, sample_num=10, file_name="cross_attn_data_rec.pt"
    ):
        if "cross_attn_list" not in dir(self):
            self.cross_attn_list = []
        cross_attn_layers = []
        for i in range(len(analysis_info_list[0]["attn_states"])):
            cross_attn_states = torch.zeros(
                (
                    len(analysis_info_list),
                    analysis_info_list[i]["attn_states"][0].size(-1),
                )
            ).cpu()
            for j, analysis_info in enumerate(analysis_info_list):
                cross_attn = analysis_info["attn_states"][i]
                cross_attn_states[j] = cross_attn[0, 0].cpu()
            cross_attn_layers.append(cross_attn_states)
        self.cross_attn_list.append(cross_attn_layers)
        if self.count == sample_num:
            torch.save(self.cross_attn_list, file_name)

    def decoder_connection_analysis(
        self,
        analysis_info_list,
        sample_num=10,
        file_name="decoder_connection_data_rec.pt",
    ):
        if "decoder_connection_states_norms_list" not in dir(self):
            self.decoder_connection_states_norms_list = []
        connection_states_norms_layers = []

        encoder_len = analysis_info_list[0]["attn_states"][0].size()[-1]
        for i in range(len(analysis_info_list[0]["connection_states"])):
            connection_states_norms_pos_list = []
            for j, analysis_info in enumerate(analysis_info_list):
                connections = analysis_info["connection_states"][i].view(
                    encoder_len + 2 + j, encoder_len + 1 + j
                )
                connections = connections[1:].cpu()

                connection_states_norms_pos_list.append(connections)
            connection_states_norms_layers.append(connection_states_norms_pos_list)
        self.decoder_connection_states_norms_list.append(connection_states_norms_layers)
        if self.count == sample_num:
            torch.save(
                self.decoder_connection_states_norms_list, file_name,
            )

    def decoder_norm_attribution(
        self,
        analysis_info_list: List[Dict],
        net_input,
        pred_tokens: LongTensor,
        positional_scores: Tensor,
        bos_token: Optional[int] = None,
        drop_rate=0.1,
    ):
        compositions_states_norms = torch.zeros(
            (
                len(analysis_info_list),
                analysis_info_list[-1]["compositions_states"][0].size(1),
            )
        )
        for j, analysis_info in enumerate(analysis_info_list):
            compositions = analysis_info["compositions_states"][-1]
            compositions_states_norms[j, : compositions.size(1)] = torch.norm(
                compositions, dim=-1
            )[0, :, 0]

        self.compute_scores(
            compositions_states_norms,
            net_input,
            pred_tokens,
            positional_scores,
            bos_token,
            drop_rate,
        )
        self.AOPC()

    def decoder_norm_logit_analysis(
        self,
        analysis_info_list,
        pred_tokens,
        sample_num=10,
        file_name="decoder_norms_logits_data_rec.pt",
    ):
        if "decoder_compositions_states_norms_list" not in dir(self):
            self.decoder_compositions_states_norms_list = []
        if "decoder_compositions_states_logit_list" not in dir(self):
            self.decoder_compositions_states_logit_list = []
        compositions_states_norms_layers = []
        compositions_states_logits_layers = []
        for i in range(len(analysis_info_list[0]["compositions_states"])):
            compositions_states_norms = torch.zeros(
                (
                    len(analysis_info_list),
                    analysis_info_list[-1]["compositions_states"][0].size(1),
                )
            ).cpu()
            compositions_states_logits = torch.zeros_like(compositions_states_norms).cpu()
            assert len(pred_tokens) == len(analysis_info_list)
            for j, analysis_info in enumerate(analysis_info_list):
                compositions = analysis_info["compositions_states"][i]
                compositions_states_norms[j, : compositions.size(1)] = torch.norm(
                    compositions, dim=-1
                ).cpu()[0, :, 0]
                token_embed = self.model.models[0].decoder.output_projection.weight[
                    pred_tokens[j]
                ]
                compositions_states_logits[j, : compositions.size(1)] = torch.matmul(
                    compositions[0, :, 0], token_embed
                ).cpu()
            compositions_states_norms_layers.append(compositions_states_norms)
            compositions_states_logits_layers.append(compositions_states_logits)
        self.decoder_compositions_states_norms_list.append(
            compositions_states_norms_layers
        )
        self.decoder_compositions_states_logit_list.append(
            compositions_states_logits_layers
        )
        if self.count == sample_num:
            torch.save(
                (
                    self.decoder_compositions_states_norms_list,
                    self.decoder_compositions_states_logit_list,
                ),
                file_name,
            )

    def decoder_norm_analysis(
        self, analysis_info_list, sample_num=10, file_name="decoder_norms_data_rec.pt"
    ):
        if "decoder_compositions_states_norms_list" not in dir(self):
            self.decoder_compositions_states_norms_list = []
        compositions_states_norms_layers = []
        for i in range(len(analysis_info_list[0]["compositions_states"])):
            compositions_states_norms = torch.zeros(
                (
                    len(analysis_info_list),
                    analysis_info_list[-1]["compositions_states"][0].size(1),
                )
            ).cpu()
            for j, analysis_info in enumerate(analysis_info_list):
                compositions = analysis_info["compositions_states"][i]
                compositions_states_norms[j, : compositions.size(1)] = torch.norm(
                    compositions, dim=-1
                ).cpu()[0, :, 0]
            compositions_states_norms_layers.append(compositions_states_norms)
        self.decoder_compositions_states_norms_list.append(
            compositions_states_norms_layers
        )
        if self.count == sample_num:
            torch.save(self.decoder_compositions_states_norms_list, file_name)

    def encoder_attn_analysis(
        self, attn_states, sample_num=10, file_name="encoder_attn_data_rec.pt"
    ):
        if "encoder_attn_list" not in dir(self):
            self.encoder_attn_list = []
        encoder_attn_states = []
        for layer_attn in attn_states:
            encoder_attn_states.append(layer_attn[0].cpu())
        self.encoder_attn_list.append(encoder_attn_states)
        if self.count == sample_num:
            torch.save(self.encoder_attn_list, file_name)

    def encoder_norm_analysis(
        self, compositions_states, sample_num=10, file_name="encoder_norms_data_rec.pt"
    ):
        compositions_states_norms = []
        for compositions in compositions_states:
            compositions_states_norms.append(
                torch.norm(compositions, dim=-1).squeeze(-1).cpu()
            )
        if "encoder_compositions_states_norms_list" not in dir(self):
            self.encoder_compositions_states_norms_list = []
        self.encoder_compositions_states_norms_list.append(compositions_states_norms)
        if self.count == sample_num:
            torch.save(self.encoder_compositions_states_norms_list, file_name)

    def encoder_connection_norm_analysis(
        self,
        connection_states,
        sample_num=10,
        file_name="encoder_connection_norms_data_rec.pt",
    ):
        connection_states_norms = []
        encoder_len = connection_states[0].size(0)
        for connection in connection_states:
            connection = connection.view(encoder_len, encoder_len + 1, encoder_len)
            connection = connection[:, 1:]
            connection_states_norms.append(connection.cpu())
        if "encoder_connection_states_norms_list" not in dir(self):
            self.encoder_connection_states_norms_list = []
        self.encoder_connection_states_norms_list.append(connection_states_norms)
        if self.count == sample_num:
            torch.save(self.encoder_connection_states_norms_list, file_name)

    def AOPC(self):
        if self.count % 100 == 0:
            temp_drop_probs = torch.stack(self.drop_probs)
            logger.info("AOPC: {}".format(torch.mean(temp_drop_probs).item()))

        # all_positional_scores = [self.positional_scores, self.isolate_positional_scores]
        if self.count == 3003 or self.count == 6750:  # or self.count > 3000:
            logger.info("end inference")
            drop_probs = torch.stack(self.drop_probs)
            analysis_info_tensor = torch.stack(self.analysis_info_list)
            logger.info("avg coef: {}".format(torch.mean(analysis_info_tensor).item()))
            logger.info("AOPC: {}".format(torch.mean(drop_probs).item()))
            # for i in range(len(self.bias_ratio)):
            #     print((self.bias_ratio[i] / self.ratio_count[i]).item())
            # for i in range(len(self.bias_ratio)):
            #     print(self.ratio_count[i])
            # print("avg:", torch.cat(self.bias_ratio, dim=0).sum()/sum(self.ratio_count))

    def bos_difference(self, analysis_info_list):
        def check_head_word(compositions_logits):
            pred_index = torch.argmax(compositions_logits.sum(dim=2), dim=-1)[0, 0].item()
            head_word = self.tgt_dict.string([pred_index])
            lower_head_word = head_word.lower()
            if lower_head_word == head_word:
                return None
            lower_word_index = self.tgt_dict.index(lower_head_word)
            if lower_word_index == self.tgt_dict.unk():
                return None
            return pred_index, lower_word_index

        compositions_logits = analysis_info_list[0]["compositions_logits"]
        res = check_head_word(compositions_logits)
        if res is not None:
            pred_index, lower_word_index = res
            wo_dif = (
                compositions_logits[0, 0, 1:-1, pred_index].sum()
                - compositions_logits[0, 0, 1:-1, lower_word_index].sum()
            )
            w_dif = (
                compositions_logits[0, 0, 1:, pred_index].sum()
                - compositions_logits[0, 0, 1:, lower_word_index].sum()
            )
            self.data_rec.append((wo_dif.item(), w_dif.item()))

        if self.count == 3003 or self.count == 6750:  # or self.count > 3000:

            wo_dif_sum = 0
            w_dif_sum = 0
            for wo_dif, w_dif in self.data_rec:
                wo_dif_sum += wo_dif
                w_dif_sum += w_dif
            torch.save(self.data_rec, "data_rec.pt")
            print("avg w/o dif", wo_dif_sum / len(self.data_rec))
            print("avg w dif", w_dif_sum / len(self.data_rec))

    def norm_analysis(self):
        if self.count == 3003 or self.count == 6750:  # or self.count > 3000:
            for sample_analysis_info in self.analysis_info_list:
                for analysis_info in sample_analysis_info:
                    analysis_info["composition_norms"] = analysis_info[
                        "composition_norms"
                    ].cpu()
            torch.save(self.analysis_info_list, "analysis.pt")

    def compute_scores(
        self,
        decomposition_positional_scores: Tensor,
        net_input,
        pred_tokens: LongTensor,
        positional_scores: Tensor,
        bos_token: Optional[int] = None,
        drop_rate=0.1,
    ):
        decomposition_positional_scores = decomposition_positional_scores[:, 1:]
        positional_probs = torch.exp(positional_scores)

        bos_token = self.eos if bos_token is None else bos_token
        prefix_tokens = torch.zeros_like(pred_tokens).to(pred_tokens)
        prefix_tokens[0] = bos_token
        prefix_tokens[1:] = pred_tokens[:-1]

        pred_len = decomposition_positional_scores.size(0)
        src_tokens = net_input["src_tokens"]

        bsz, src_len = src_tokens.size()[:2]
        assert bsz == 1
        pad_num = (src_tokens[0] == self.tgt_dict.pad()).sum()
        assert pad_num == 0

        assert src_len + pred_len == decomposition_positional_scores.size(1)

        sample_drop_probs = []

        for i in range(pred_len):
            tokens_to_keep = max(int(round((src_len + i + 1) * drop_rate)), 1)
            _, top_indices = torch.topk(
                decomposition_positional_scores[i, : src_len + i + 1], k=tokens_to_keep
            )
            encoder_dropout_tokens = top_indices[top_indices < src_len].unsqueeze(dim=0)
            decoder_dropout_tokens = top_indices[~(top_indices < src_len)].unsqueeze(
                dim=0
            )
            if encoder_dropout_tokens.size(1) == 0:
                encoder_dropout_tokens = None
            if decoder_dropout_tokens.size(1) == 0:
                decoder_dropout_tokens = None
            else:
                decoder_dropout_tokens -= src_len
            assert (
                encoder_dropout_tokens is not None or decoder_dropout_tokens is not None
            )
            net_input["return_compositions"] = False
            net_input["dropout_tokens"] = encoder_dropout_tokens
            encoder_outs = self.model.forward_encoder(net_input)
            lprobs, avg_attn_scores, _, _ = self.model.forward_decoder(
                prefix_tokens[None, : i + 1],
                encoder_outs,
                incremental_states=[None],
                temperature=self.temperature,
                return_compositions=False,
                dropout_tokens=decoder_dropout_tokens,
            )

            sample_drop_probs.append(
                positional_probs[i] - torch.exp(lprobs[0, pred_tokens[i]])
            )

        self.drop_probs.append(torch.stack(sample_drop_probs).mean())

    def _prefix_tokens(
        self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                :, 0, 1 : step + 1
            ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        src_lengths,
        max_len: int,
        decomposition_eos_scores,
        decomposition_scores,
        src_len,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]

        # decomposition_pos_scores = decomposition_scores.index_select(0, bbsz_idx)[
        #     :, : step + 1, : src_len + step + 2
        # ]

        pos_scores[:, step] = eos_scores

        # decomposition_pos_scores[:, step] = decomposition_eos_scores

        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        # The keys here are of the form "{sent}_{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # set() is not supported in script export
        sents_seen: Dict[str, Optional[Tensor]] = {}

        # For every finished beam item
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            # sentence index in the current (possibly reduced) batch
            unfin_idx = idx // beam_size
            # sentence index in the original (unreduced) batch
            sent = unfin_idx + cum_unfin[unfin_idx]
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string
            seen = str(sent.item()) + "_" + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None

            if self.match_source_len and step > src_lengths[unfin_idx]:
                score = torch.tensor(-math.inf).to(score)

            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": score,
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                        "decomposition_positional_scores": None,  # decomposition_pos_scores[i],
                    }
                )

        newly_finished: List[int] = []

        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))

            if not finished[sent] and self.is_finished(
                step, unfin_idx, max_len, len(finalized[sent]), beam_size
            ):
                finished[sent] = True
                newly_finished.append(unfin_idx)

        return newly_finished

    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False

    def calculate_banned_tokens(
        self,
        tokens,
        step: int,
        gen_ngrams: List[Dict[str, List[int]]],
        no_repeat_ngram_size: int,
        bbsz_idx: int,
    ):
        tokens_list: List[int] = tokens[
            bbsz_idx, step + 2 - no_repeat_ngram_size : step + 1
        ].tolist()
        # before decoding the next token, prevent decoding of ngrams that have already appeared
        ngram_index = ",".join([str(x) for x in tokens_list])
        return gen_ngrams[bbsz_idx].get(ngram_index, torch.jit.annotate(List[int], []))

    def transpose_list(self, l: List[List[int]]):
        # GeneratorExp aren't supported in TS so ignoring the lint
        min_len = min([len(x) for x in l])  # noqa
        l2 = [[row[i] for row in l] for i in range(min_len)]
        return l2

    def _no_repeat_ngram(self, tokens, lprobs, bsz: int, beam_size: int, step: int):
        # for each beam and batch sentence, generate a list of previous ngrams
        gen_ngrams: List[Dict[str, List[int]]] = [
            torch.jit.annotate(Dict[str, List[int]], {})
            for bbsz_idx in range(bsz * beam_size)
        ]
        cpu_tokens = tokens.cpu()
        for bbsz_idx in range(bsz * beam_size):
            gen_tokens: List[int] = cpu_tokens[bbsz_idx].tolist()
            for ngram in self.transpose_list(
                [gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]
            ):
                key = ",".join([str(x) for x in ngram[:-1]])
                gen_ngrams[bbsz_idx][key] = gen_ngrams[bbsz_idx].get(
                    key, torch.jit.annotate(List[int], [])
                ) + [ngram[-1]]

        if step + 2 - self.no_repeat_ngram_size >= 0:
            # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            banned_tokens = [
                self.calculate_banned_tokens(
                    tokens, step, gen_ngrams, self.no_repeat_ngram_size, bbsz_idx
                )
                for bbsz_idx in range(bsz * beam_size)
            ]
        else:
            banned_tokens = [
                torch.jit.annotate(List[int], []) for bbsz_idx in range(bsz * beam_size)
            ]
        for bbsz_idx in range(bsz * beam_size):
            lprobs[bbsz_idx][torch.tensor(banned_tokens[bbsz_idx]).long()] = torch.tensor(
                -math.inf
            ).to(lprobs)
        return lprobs


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        self.has_incremental: bool = False
        if all(
            hasattr(m, "decoder") and isinstance(m.decoder, FairseqIncrementalDecoder)
            for m in models
        ):
            self.has_incremental = True

    def forward(self):
        pass

    def has_encoder(self):
        return hasattr(self.single_model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min([m.max_decoder_positions() for m in self.models])

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_decoder(
        self,
        tokens,
        encoder_outs: List[Dict[str, List[Tensor]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
        return_compositions: bool = False,
        dropout_tokens: Tensor = None,
        return_connection: int = None,
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                    return_compositions=return_compositions,
                    dropout_tokens=dropout_tokens,
                    return_connection=return_connection,
                )
            else:
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    return_compositions=return_compositions,
                    dropout_tokens=dropout_tokens,
                    return_connection=return_connection,
                )

            attn: Optional[Tensor] = None
            compositions_logits = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                compositions_logits = decoder_out[1]["compositions_logits"]
                analysis_info = decoder_out[1]["analysis_info"]
                if analysis_info is not None:
                    analysis_info["attn_states"] = decoder_out[1]["attn_states"]
                    analysis_info["connection_states"] = decoder_out[1][
                        "connection_states"
                    ]
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )

            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            probs = probs[:, -1, :]

            if return_compositions:
                compositions_logits = compositions_logits[:, -1, :, :]
                compositions_logits = compositions_logits.transpose(
                    1, 2
                ).contiguous()  # bsz*beam x dict_size x src_len+gen_len

            if self.models_size == 1:
                return probs, attn, compositions_logits, analysis_info

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def reorder_encoder_out(
        self, encoder_outs: Optional[List[Dict[str, List[Tensor]]]], new_order
    ):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[Dict[str, List[Tensor]]] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(model.encoder.reorder_encoder_out(encoder_outs[i], new_order))
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(
        self, incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]], new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order
            )


class SequenceGeneratorWithAlignment(SequenceGenerator):
    def __init__(self, models, tgt_dict, left_pad_target=False, **kwargs):
        """Generates translations of a given source sentence.

        Produces alignments following "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            left_pad_target (bool, optional): Whether or not the
                hypothesis should be left padded or not when they are
                teacher forced for generating alignments.
        """
        super().__init__(EnsembleModelWithAlignment(models), tgt_dict, **kwargs)
        self.left_pad_target = left_pad_target

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        finalized = super()._generate(sample, **kwargs)

        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        (
            src_tokens,
            src_lengths,
            prev_output_tokens,
            tgt_tokens,
        ) = self._prepare_batch_for_alignment(sample, finalized)
        if any(getattr(m, "full_context_alignment", False) for m in self.model.models):
            attn = self.model.forward_align(src_tokens, src_lengths, prev_output_tokens)
        else:
            attn = [
                finalized[i // beam_size][i % beam_size]["attention"].transpose(1, 0)
                for i in range(bsz * beam_size)
            ]

        if src_tokens.device != "cpu":
            src_tokens = src_tokens.to("cpu")
            tgt_tokens = tgt_tokens.to("cpu")
            attn = [i.to("cpu") for i in attn]

        # Process the attn matrix to extract hard alignments.
        for i in range(bsz * beam_size):
            alignment = utils.extract_hard_alignment(
                attn[i], src_tokens[i], tgt_tokens[i], self.pad, self.eos
            )
            finalized[i // beam_size][i % beam_size]["alignment"] = alignment
        return finalized

    def _prepare_batch_for_alignment(self, sample, hypothesis):
        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        src_tokens = (
            src_tokens[:, None, :]
            .expand(-1, self.beam_size, -1)
            .contiguous()
            .view(bsz * self.beam_size, -1)
        )
        src_lengths = sample["net_input"]["src_lengths"]
        src_lengths = (
            src_lengths[:, None]
            .expand(-1, self.beam_size)
            .contiguous()
            .view(bsz * self.beam_size)
        )
        prev_output_tokens = data_utils.collate_tokens(
            [beam["tokens"] for example in hypothesis for beam in example],
            self.pad,
            self.eos,
            self.left_pad_target,
            move_eos_to_beginning=True,
        )
        tgt_tokens = data_utils.collate_tokens(
            [beam["tokens"] for example in hypothesis for beam in example],
            self.pad,
            self.eos,
            self.left_pad_target,
            move_eos_to_beginning=False,
        )
        return src_tokens, src_lengths, prev_output_tokens, tgt_tokens


class EnsembleModelWithAlignment(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    def forward_align(self, src_tokens, src_lengths, prev_output_tokens):
        avg_attn = None
        for model in self.models:
            decoder_out = model(src_tokens, src_lengths, prev_output_tokens)
            attn = decoder_out[1]["attn"][0]
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_attn.div_(len(self.models))
        return avg_attn
