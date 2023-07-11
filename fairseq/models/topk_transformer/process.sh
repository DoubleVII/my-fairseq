TEXT=E:/code/data/nmt/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir E:/code/data/nmt/data-bin/iwslt14.tokenized.de-en_joined_test_dict \
    --workers 2 \
    --srcdict E:/code/data/nmt/data-bin/iwslt14.tokenized.de-en/dict.de.txt \
    --tgtdict E:/code/data/nmt/data-bin/iwslt14.tokenized.de-en/dict.en.txt

    # --joined-dictionary \
# TEXT=~/data/wmt14
# fairseq-preprocess --source-lang de --target-lang en \
#     --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#     --destdir data-bin/wmt14_joined \
#     --workers 20 \
#     --joined-dictionary