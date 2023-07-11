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



fairseq-train \
   E:/code/data/nmt/data-bin/iwslt14.tokenized.de-en \
   --arch dldc_transformer_arch --share-decoder-input-output-embed \
   --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
   --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
   --dropout 0.3 --weight-decay 0.0001 \
   --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
   --max-tokens 100 \
   --eval-bleu \
   --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
   --eval-bleu-detok moses \
   --eval-bleu-remove-bpe \
   --eval-bleu-print-samples \
   --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
   --save-interval 1 \
   --keep-best-checkpoints 1 \
   --update-freq 2 \
   --attention-dropout 0.3 \
   --no-epoch-checkpoints \
   --encoder-ffn-embed-dim 1024 \
   --encoder-embed-dim 256 \
   --decoder-ffn-embed-dim 1024 \
   --decoder-embed-dim 256 \
   --encoder-attention-heads 8 \
   --decoder-attention-heads 8 \
   --encoder-layers 4 \
   --decoder-layers 4 \
   --encoder-normalize-before \
   --decoder-normalize-before \
   --cpu

# 8128512

# fairseq-train \
#    E:/code/data/nmt/data-bin/iwslt14.tokenized.de-en \
#    --arch layer_attention_transformer_arch --share-all-embeddings \
#    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#    --dropout 0.3 --weight-decay 0.0001 \
#    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#    --max-tokens 4096 \
#    --eval-bleu \
#    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#    --eval-bleu-detok moses \
#    --eval-bleu-remove-bpe \
#    --eval-bleu-print-samples \
#    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#    --save-interval 1 \
#    --keep-best-checkpoints 1 \
#    --update-freq 2 \
#    --attention-dropout 0.3 \
#    --layer-attn-reduction mean \
#    --no-epoch-checkpoints \
#    --fp16 \
#    --encoder-ffn-embed-dim 1024 \
#    --encoder-embed-dim 256 \
#    --decoder-ffn-embed-dim 1024 \
#    --decoder-embed-dim 256 \
#    --encoder-attention-heads 8 \
#    --decoder-attention-heads 8 \
#    --encoder-layers 2 \
#    --decoder-layers 2 \
#    --encoder-recurrent 3 \
#    --decoder-recurrent 3 \
#    --layer-aggregation attn \
#    --cpu
#    --only-last-recurrent \
#    --no-layer-attn



# TEXT=~/data/iwslt14.tokenized.de-en
# fairseq-preprocess --source-lang de --target-lang en \
#     --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#     --destdir data-bin/iwslt14.tokenized.de-en_joined \
#     --workers 20 \
#     --joined-dictionary

TEXT=~/data/wmt14
fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt14_joined_en_de \
    --workers 20 \
    --joined-dictionary


# CUDA_VISIBLE_DEVICES=1,2,3,4,5 fairseq-train \
#     ~/data/data-bin/iwslt14.tokenized.de-en_joined \
#     --arch layer_attention_transformer_arch --share-all-embeddings \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#     --dropout 0.35 --weight-decay 0.0001 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 800 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#     --save-interval 5 \
#     --keep-best-checkpoints 1 \
#     --fp16 \
#     --update-freq 4
