echo "Convert wavs at $1/$2 into mels at $1/$3 with $4 channels"

R_PATH=$1
WAV_PATH=$(echo $2 | sed -e "s:\/:\\\/:g")
MEL_PATH=$(echo $3 | sed -e "s:\/:\\\/:g")


ls $1$2 | sed -e "s:\(.*\):$WAV_PATH\1|None:" > $1/waveglow_preprocess_metadata.csv
ls $1$2 | sed -e "s:\([^.]*\).*:$MEL_PATH\1.mel|None:" > $1/mel_metadata.csv

python ~/ml/tacotron2/preprocess_audio2mel.py \
    --dataset-path=$1 \
    --wav-files=$1/waveglow_preprocess_metadata.csv \
    --mel-files=$1/mel_metadata.csv \
    --n-mel-channels=$4 \
