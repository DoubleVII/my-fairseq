# fairseq-generate ~/data/data-bin/iwslt14.tokenized.de-en_joined \
#     --path checkpoints/checkpoint_best.pt \
#     --batch-size 128 --beam 5 --remove-bpe \
#     --scoring bleu \
#     --quiet

fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
