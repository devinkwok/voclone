## incomplete list of commands for processing labelled data

BASEDIR=$1
NAME=$2
./resample-wavs.sh $BASEDIR/wav/$NAME $BASEDIR/norm-wav/$NAME
## rename to remove mp3, etc.

mkdir $BASEDIR/roughsplitwav/$NAME
python make_intervals.py --operation generate \
    --input_dir $BASEDIR/norm-wav/$NAME \
    --output_dir $BASEDIR/roughsplitwav/$NAME \
    --threshold 0.0002

python make_intervals.py --operation split \
    --input_dir $BASEDIR/norm-wav/$NAME \
    --output_dir $BASEDIR/roughsplitwav/$NAME \

mkdir $BASEDIR/mel160/$NAME/
./wav2mel.sh $BASEDIR/ roughsplitwav/$NAME/ mel160/$NAME/ 160


## incomplete list of commands for classifying unlabelled data
START_DIR=~/ml/voclone-tools
DIS_DIR=UGATIT-from-gtx960/
R_DIR=#TODO

./resample-wavs.sh $BASEDIR/wav/$NAME $BASEDIR/norm-wav/$NAME
## rename to remove mp3, etc.

#TODO need to copy and append -voice to filenames
# this is to remove noise
mkdir $BASEDIR/roughsplitwav/$NAME
python make_intervals.py --operation generate \
    --input_dir $BASEDIR/norm-wav/$NAME \
    --output_dir $BASEDIR/roughsplitwav/$NAME \
    --threshold 0.0002

python make_intervals.py --operation split \
    --input_dir $BASEDIR/norm-wav/$NAME \
    --output_dir $BASEDIR/roughsplitwav/$NAME \

# keep the silent parts for later
mv $BASEDIR/roughsplitwav/$NAME/noise $BASEDIR/roughsplitwav/$NAME/silence-noise

# use the non-silent parts to classify
mkdir $BASEDIR/norm-wav/$NAME-ns
mv $BASEDIR/roughsplitwav/$NAME/target/* $BASEDIR/norm-wav/$NAME-ns/
mkdir $BASEDIR/mel-classify/$NAME
./wav2mel.sh $BASEDIR/ roughsplitwav/$NAME-ns/ mel-classify/$NAME/ 80
rm $BASEDIR/roughsplitwav/$NAME/source -dr
rm $BASEDIR/roughsplitwav/$NAME/silence -dr
cd $DIS_DIR/dataset/classify
rm *
ln -s $BASEDIR/mel-classify/$NAME $NAME
cd $DIS_DIR
./train-mel.sh

mv $DIS_DIR/results/classify/ugatit-dis-outputs.csv $R_DIR/ugatit-dis-outputs.csv
cd $R_DIR
Rscript ugatit-linear-classifier.R
mv $R_DIR/predicted-ugatit-dis-outputs.csv $BASEDIR/roughsplitwav/$NAME/

#TODO need to remove / from before filenames in csv
# turn the predictions into intervals and split audio

# do this again with lower threshold to get noise
cd $START_DIR
python make_intervals.py --operation classify \
    --input_dir $BASEDIR/roughsplitwav/$NAME/predicted-ugatit-dis-outputs.csv \
    --output_dir $BASEDIR/roughsplitwav/$NAME/intervals-source.csv \
    --threshold 0.7

python make_intervals.py --operation split \
    --input_dir $BASEDIR/norm-wav/$NAME-ns \
    --output_dir $BASEDIR/roughsplitwav/$NAME \

# check over the split
# make mels
mkdir $BASEDIR/mel160/$NAME/noise/
mkdir $BASEDIR/mel160/$NAME/trainA/
./wav2mel.sh $BASEDIR/ roughsplitwav/$NAME/silence-noise/ mel160/$NAME/noise/ 160
./wav2mel.sh $BASEDIR/ roughsplitwav/$NAME/target/ mel160/$NAME/trainA/ 160

# rename and put in dataset