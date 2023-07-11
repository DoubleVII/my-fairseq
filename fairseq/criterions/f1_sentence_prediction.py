import fairseq
from fairseq.criterions import register_criterion
from fairseq.criterions.sentence_prediction import SentencePredictionCriterion
import torch.nn.functional as F
from fairseq import metrics, utils
import torch
import math

# from fairseq.modules import FairseqDropout
all_pred = []
all_target = []

all_score = []


@register_criterion("f1_sentence_prediction")
class F1SentencePredictionCriterion(SentencePredictionCriterion):
    def forward(self, model, sample, reduce=True):
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

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduction="sum")
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(logits, targets, reduction="sum")

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        if not self.regression_target:
            preds = logits.argmax(dim=1)
            pred_score = F.softmax(logits, dim=1)
            logging_output["ncorrect"] = (preds == targets).sum()

            global all_score

            if not model.training:
                all_pred.extend(preds.tolist())
                all_target.extend(targets.tolist())
                all_score.append(pred_score)

                if len(all_pred) == 57544:  # 113000:
                    with open("/home/yangs/data/assignment2/pred_file", "w") as pred_file:
                        pred_file.write(" ".join(map(str, all_pred)))
                    with open(
                        "/home/yangs/data/assignment2/target_file", "w"
                    ) as target_file:
                        target_file.write(" ".join(map(str, all_target)))
                    all_score = torch.cat(all_score, dim=0)

                    with open(
                        "/home/yangs/data/assignment2/score_file", "wb"
                    ) as score_file:
                        torch.save(all_score, score_file)

            # TP0 = (
            #     (preds == targets)
            #     .to(targets)
            #     .masked_fill((targets == 1) | (targets == 2), 0)
            #     .sum()
            # )

            # FN0 = (targets == 0).sum() - TP0
            # FP0 = (preds == 0).sum() - TP0
            # if FN0 < 0:
            #     print(preds)
            #     print(targets)
            #     print(TP0, FN0)

            # assert FN0 >= 0
            # assert FP0 >= 0

            # TP1 = (
            #     (preds == targets)
            #     .to(targets)
            #     .masked_fill((targets == 0) | (targets == 2), 0)
            #     .sum()
            # )

            # FN1 = (targets == 1).sum() - TP1
            # FP1 = (preds == 1).sum() - TP1
            # assert FN1 >= 0
            # assert FP1 >= 0

            # TP2 = (
            #     (preds == targets)
            #     .to(targets)
            #     .masked_fill((targets == 0) | (targets == 1), 0)
            #     .sum()
            # )

            # FN2 = (targets == 2).sum() - TP2
            # FP2 = (preds == 2).sum() - TP2
            # assert FN2 >= 0
            # assert FP2 >= 0

            # logging_output["TP0"] = TP0
            # logging_output["TP1"] = TP1
            # logging_output["TP2"] = TP2
            # logging_output["FN0"] = FN0
            # logging_output["FN1"] = FN1
            # logging_output["FN2"] = FN2
            # logging_output["FP0"] = FP0
            # logging_output["FP1"] = FP1
            # logging_output["FP2"] = FP2

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )

        # if len(logging_outputs) > 0 and "TP0" in logging_outputs[0]:
        #     TP0 = sum(log.get("TP0", 0) for log in logging_outputs)
        #     TP1 = sum(log.get("TP1", 0) for log in logging_outputs)
        #     TP2 = sum(log.get("TP2", 0) for log in logging_outputs)
        #     FN0 = sum(log.get("FN0", 0) for log in logging_outputs)
        #     FN1 = sum(log.get("FN1", 0) for log in logging_outputs)
        #     FN2 = sum(log.get("FN2", 0) for log in logging_outputs)
        #     FP0 = sum(log.get("FP0", 0) for log in logging_outputs)
        #     FP1 = sum(log.get("FP1", 0) for log in logging_outputs)
        #     FP2 = sum(log.get("FP2", 0) for log in logging_outputs)
        #     # print(TP0, TP1, TP2, FN0, FN1, FN2, FP0, FP1, FP2)
        #     precision0 = TP0 // (TP0 + FP0)
        #     precision1 = TP1 // (TP1 + FP1)
        #     precision2 = TP2 // (TP2 + FP2)
        #     if torch.isnan(precision0):
        #         precision0 = torch.FloatTensor(0).to(precision0)
        #     if torch.isnan(precision1):
        #         precision1 = torch.FloatTensor(0).to(precision1)
        #     if torch.isnan(precision0):
        #         precision2 = torch.FloatTensor(0).to(precision2)
        #     recall0 = TP0 // (TP0 + FN0)
        #     recall1 = TP1 // (TP1 + FN1)
        #     recall2 = TP2 // (TP2 + FN2)
        #     macro_p = (precision0 + precision1 + precision2) // 3
        #     macro_r = (recall0 + recall1 + recall2) // 3
        #     # print(macro_p)
        #     # print(macro_r)
        #     macro_f1 = 2 * macro_p * macro_r // (macro_p + macro_r)
        #     metrics.log_scalar("macro_f1", 100.0 * macro_f1, nsentences, round=1)

