fairseq-train \
    E:/code/data/nmt/data-bin/iwslt14.tokenized.de-en \
    --arch embedded_encoder_transformer_arch --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-tokens 100 \
    --no-epoch-checkpoints \
    --encoder-ffn-embed-dim 1024 \
    --encoder-embed-dim 256 \
    --decoder-ffn-embed-dim 1024 \
    --decoder-embed-dim 256 \
    --encoder-attention-heads 8 \
    --decoder-attention-heads 8 \
    --save-interval-updates 200 \
    --cpu


# fairseq-train \
#     E:/code/data/nmt/data-bin/iwslt14.tokenized.de-en \
#     --arch my_transformer_arch --share-decoder-input-output-embed \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 100 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#     --cpu

# fairseq-train \
#     E:/code/data/nmt/data-bin/iwslt14.tokenized.de-en \
#     --arch layer_attention_transformer_arch --share-decoder-input-output-embed \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 100 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#     --cpu


# TEXT=~/data/iwslt14.tokenized.de-en
# fairseq-preprocess --source-lang de --target-lang en \
#     --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#     --destdir data-bin/iwslt14.tokenized.de-en_joined \
#     --workers 20 \
#     --joined-dictionary

# TEXT=~/data/wmt14
# fairseq-preprocess --source-lang de --target-lang en \
#     --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#     --destdir data-bin/wmt14_joined \
#     --workers 20 \
#     --joined-dictionary


# fairseq-train \
#     --arch example_arch \
#     --task example_task \
#     --up-bound 100 \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 30 \
#     --weight-decay 0.0001 \
#     --criterion example_criterion \
#     --max-tokens 100 \
#     --batch-size 40000 \
#     --layer-num 1 \
#     --linear-dim 3 \
#     --no-epoch-checkpoints \
#     --cpu