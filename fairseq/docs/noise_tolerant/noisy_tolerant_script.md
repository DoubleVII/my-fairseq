TEXT=wmt16_en_de_bpe32k_filtered
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/train.tok.clean.bpe.32000 \
  --validpref $TEXT/newstest2013.tok.bpe.32000 \
  --testpref $TEXT/newstest2014.tok.bpe.32000 \
  --destdir data-bin/wmt16_en_de_bpe32k_filtered \
  --nwordssrc 32768 --nwordstgt 32768 \
  --joined-dictionary \
  --workers 20


TEXT=sample_wmt16_en_de_bpe32k
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/train.tok.clean.bpe.32000 \
  --validpref $TEXT/newstest2013.tok.bpe.32000 \
  --testpref $TEXT/newstest2014.tok.bpe.32000 \
  --destdir data-bin/sample_wmt16_en_de_bpe32k \
  --nwordssrc 32768 --nwordstgt 32768 \
  --joined-dictionary \
  --workers 20


1970000,2410000d

TOTAL_UPDATES=10    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x

DATA_DIR=/home/yangs/data-bin/wmt16_en_de_bpe32k_filter

<!-- fairseq-train --fp16 $DATA_DIR \
    --task concatenate_masked_lm  --criterion masked_lm \
    --arch roberta_base \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 -->

TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x

DATA_DIR=/home/yangs/data/data-bin/wmt16_en_de_bpe32k_filter
fairseq-train --fp16 $DATA_DIR \
    --task concatenate_masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode eos --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 \
    --valid-subset test

TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x
DATA_DIR=/home/yangs/data/data-bin/wmt16_en_de_bpe32k_filter

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_DIR --fp16 \
    --task concatenate_masked_lm --criterion masked_lm_sentence_prediction \
    --arch multi_task_roberta_small --sample-break-mode eos --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 20 \
    --valid-subset test \
    --num-classes 2



TEXT=sb_to_put
fairseq-preprocess --source-lang src --target-lang tgt \
  --trainpref $TEXT/train_trans.char2word.tok \
  --validpref $TEXT/dev_trans.char2word.tok \
  --destdir data-bin/chinese_text_correction_char2word \
  --nwordssrc 5799 --nwordstgt 30000 \
  --workers 20


TEXT=sb_to_put
fairseq-preprocess --source-lang src --target-lang tgt \
  --testpref $TEXT/test_trans.tok \
  --destdir data-bin/chinese_text_correction_test \
  --srcdict data-bin/chinese_text_correction/dict.src.txt \
  --joined-dictionary \
  --workers 20



TEXT=iwslt14_IRS_UU
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/train.near.cat \
  --validpref $TEXT/valid \
  --testpref $TEXT/test \
  --destdir data-bin/iwslt14_IRS_UU_near \
  --nwordssrc 10000 \
  --joined-dictionary \
  --workers 20

TEXT=iwslt14_IRS_UU
fairseq-preprocess --source-lang de --target-lang en \
  --testpref $TEXT/wmt_test_prefix \
  --destdir data-bin/iwslt14_IRS_test_wmt_prefix4UUnear \
  --srcdict data-bin/iwslt14_IRS_UU_near/dict.de.txt \
  --joined-dictionary \
  --workers 20

TEXT=iwslt14_IRS_UU
fairseq-preprocess --source-lang de --target-lang en \
  --testpref $TEXT/wmt_test_prefix \
  --destdir data-bin/iwslt14_IRS_test_wmt_prefix4base \
  --srcdict data-bin/iwslt14_base/dict.de.txt \
  --joined-dictionary \
  --workers 20

TEXT=iwslt14_IRS_UU
fairseq-preprocess --source-lang de --target-lang en \
  --testpref $TEXT/wmt_test_prefix \
  --destdir data-bin/iwslt14_IRS_test_wmt_prefix4UU \
  --srcdict data-bin/iwslt14_IRS_UU/dict.de.txt \
  --joined-dictionary \
  --workers 20

TEXT=iwslt14_IRS_UU
fairseq-preprocess --source-lang de --target-lang en \
  --testpref $TEXT/wmt_test \
  --destdir data-bin/iwslt14_IRS_test_wmt4UU \
  --srcdict data-bin/iwslt14_IRS_UU/dict.de.txt \
  --joined-dictionary \
  --workers 20

fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/train.cat \
  --validpref $TEXT/valid \
  --testpref $TEXT/test \
  --destdir data-bin/iwslt14_IRS_UUnear_UR \
  --nwordssrc 10000 \
  --joined-dictionary \
  --workers 20


TEXT=sb_to_put
fairseq-preprocess --source-lang src --target-lang tgt \
  --trainpref $TEXT/train_trans_aug.char2word.tok \
  --validpref $TEXT/aug_char2word_dev.tok \
  --testpref $TEXT/test_trans_aug.char2word.tok \
  --destdir data-bin/chinese_text_correction_char2word_finetune \
  --srcdict data-bin/chinese_text_correction_char2word_pretrain/dict.src.txt \
  --tgtdict data-bin/chinese_text_correction_char2word_pretrain/dict.tgt.txt \
  --workers 20


TEXT=iwslt14_IRS_UU_large
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/train.cat \
  --testpref $TEXT/test \
  --destdir data-bin/iwslt14_IRS_UU_near_large \
  --nwordssrc 10000 \
  --joined-dictionary \
  --workers 20


TEXT=iwslt14_IRS_UU
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/train.near.cat \
  --testpref $TEXT/test \
  --destdir data-bin/iwslt14_IRS_UU_near_base_dict \
  --srcdict data-bin/iwslt14_base/dict.en.txt \
  --joined-dictionary \
  --workers 20