SCRIPTS=~/code/mosesdecoder/scripts
LC=$SCRIPTS/tokenizer/lowercase.perl
BPEROOT=~/code/subword-nmt/subword_nmt

INPUT_DIR=~/data/wmt14
OUTPUT_DIR=~/data/wmt14
BPE_FILE=~/data/wmt14/code

BPE_TOKENS=32000

perl $LC < $INPUT_DIR/newstest2014.tok.en > $INPUT_DIR/newstest2014.en
perl $LC < $INPUT_DIR/newstest2014.tok.de > $INPUT_DIR/newstest2014.de
perl $LC < $INPUT_DIR/newstest2013.tok.en > $INPUT_DIR/newstest2013.en
perl $LC < $INPUT_DIR/newstest2013.tok.de > $INPUT_DIR/newstest2013.de
perl $LC < $INPUT_DIR/train.tok.clean.en > $INPUT_DIR/train.clean.en
perl $LC < $INPUT_DIR/train.tok.clean.de > $INPUT_DIR/train.clean.de

TRAIN=$INPUT_DIR/train.en-de
BPE_CODE=$BPE_FILE
rm -f $TRAIN
cat $INPUT_DIR/train.clean.en >> $TRAIN
cat $INPUT_DIR/train.clean.de >> $TRAIN

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE


python $BPEROOT/apply_bpe.py -c $BPE_FILE < $INPUT_DIR/newstest2014.en > $OUTPUT_DIR/test.en
python $BPEROOT/apply_bpe.py -c $BPE_FILE < $INPUT_DIR/newstest2014.de > $OUTPUT_DIR/test.de
python $BPEROOT/apply_bpe.py -c $BPE_FILE < $INPUT_DIR/newstest2013.en > $OUTPUT_DIR/valid.en
python $BPEROOT/apply_bpe.py -c $BPE_FILE < $INPUT_DIR/newstest2013.de > $OUTPUT_DIR/valid.de
python $BPEROOT/apply_bpe.py -c $BPE_FILE < $INPUT_DIR/train.clean.en > $OUTPUT_DIR/train.en
python $BPEROOT/apply_bpe.py -c $BPE_FILE < $INPUT_DIR/train.clean.de > $OUTPUT_DIR/train.de

BPEROOT=~/code/subword-nmt/subword_nmt
python $BPEROOT/apply_bpe.py -c code < UU.near.large.de > UU.near.large.bpe.de
