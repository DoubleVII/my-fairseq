TEXT=wmt16_en_de_bpe32k_noise
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/train.tok.clean.bpe.32000.noise \
  --validpref $TEXT/newstest2013.tok.bpe.32000 \
  --testpref $TEXT/newstest2014.tok.bpe.32000 \
  --destdir data-bin/wmt16_en_de_bpe32k_noise \
  --nwordssrc 32768 --nwordstgt 32768 \
  --joined-dictionary \
  --workers 20

TEXT=tem_wmt16
fairseq-preprocess \
  --only-source \
  --validpref $TEXT/train.tok.clean.bpe.32000.de \
  --destdir data-bin/wmt16_en_de_bpe32k_tem/input1 \
  --srcdict data-bin/wmt16_en_de_bpe32k/dict.de.txt \
  --workers 20

TEXT=iwslt14_IRS_UU
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/train.cat \
  --validpref $TEXT/valid \
  --testpref $TEXT/test \
  --destdir data-bin/iwslt14_IRS_UU \
  --nwordssrc 10000 \
  --joined-dictionary \
  --workers 20


TEXT=sb_to_put
fairseq-preprocess --source-lang src --target-lang tgt \
  --trainpref $TEXT/aug_char2word.tok \
  --validpref $TEXT/aug_char2word_dev.tok \
  --destdir data-bin/chinese_text_correction_char2word_pretrain \
  --nwordssrc 5802 \
  --nwordstgt 60000 \
  --workers 20
