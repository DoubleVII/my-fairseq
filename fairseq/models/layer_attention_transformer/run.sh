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
   --arch layer_attention_transformer_arch --share-decoder-input-output-embed \
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
   --layer-attn-reduction mean \
   --no-epoch-checkpoints \
   --encoder-ffn-embed-dim 1024 \
   --encoder-embed-dim 256 \
   --decoder-ffn-embed-dim 1024 \
   --decoder-embed-dim 256 \
   --encoder-attention-heads 8 \
   --decoder-attention-heads 8 \
   --encoder-layers 2 \
   --decoder-layers 2 \
   --encoder-recurrent 3 \
   --decoder-recurrent 3 \
   --layer-aggregation none \
   --encoder-layer-route time-wise-ind \
   --decoder-layer-route time-wise-ind  \
   --time-wise-layer-attn \
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

# TEXT=~/data/wmt14
# fairseq-preprocess --source-lang en --target-lang de \
#     --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#     --destdir data-bin/wmt14_joined_en_de \
#     --workers 20 \
#     --joined-dictionary


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


# TEXT=~/data/wmt16_en_de_bpe32k
# TEXT_SUB=~/data/wmt16_en_de_bpe32k_subset
# fairseq-preprocess --source-lang en --target-lang de \
#     --trainpref $TEXT_SUB/train.tok.clean.bpe.32000.sentences.1280k \
#     --testpref $TEXT/newstest2014.tok.bpe.32000 \
#     --destdir data-bin/wmt16_en_de_bpe32k_subset_1280k \
#     --workers 20 \
#     --srcdict data-bin/wmt16_en_de_bpe32k/dict.en.txt \
#     --tgtdict data-bin/wmt16_en_de_bpe32k/dict.de.txt


TEXT=~/data/wmt16_filtered
DEST=~/data/data-bin/wmt16_splite_sentences_all
fairseq-preprocess --source-lang en --target-lang de \
    --testpref $TEXT/newstest2014.tok.bpe.32000.all \
    --destdir $DEST \
    --workers 20 \
    --srcdict data-bin/wmt16_en_de_bpe32k/dict.en.txt \
    --tgtdict data-bin/wmt16_en_de_bpe32k/dict.de.txt

TEXT=~/data/iwslt14.tokenized.de-en_domain/tmp
DEST=~/data/data-bin/iwslt14.tokenized.en_de_domain_test_raw
fairseq-preprocess --source-lang en --target-lang de \
    --testpref $TEXT/test \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --destdir $DEST \
    --workers 20 \
    --srcdict data-bin/wmt16_en_de_bpe32k/dict.en.txt \
    --tgtdict data-bin/wmt16_en_de_bpe32k/dict.de.txt


TEXT=~/data/iwslt14.tokenized.de-en_domain
DEST=~/data/data-bin/iwslt14.tokenized.en_de_domain
fairseq-preprocess --source-lang en --target-lang de \
    --testpref $TEXT/test \
    --validpref $TEXT/valid \
    --trainpref $TEXT/train \
    --destdir $DEST \
    --workers 20 \
    --srcdict data-bin/wmt16_en_de_bpe32k/dict.en.txt \
    --tgtdict data-bin/wmt16_en_de_bpe32k/dict.de.txt


and that's an idea that, if you think about it, can only fulfill you with hope.