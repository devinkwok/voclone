#!/bin/bash

echo "Resample .wav files in subdirectories of $1 and output to $2"

# need to reduce to mono first
for file in $1/*.{mp3,m4a,webm}
do
filename=$(basename -- "$file")
ffmpeg -hide_banner -loglevel warning -nostats -i "$file" -ac 1 "$1/$filename.wav"
# this is old version which doesn't normalize
# ffmpeg -hide_banner -loglevel warning -nostats -i "$file" -ar 22050 -ac 1 "$2/$filename.wav"
done
# need to `pip install ffmpeg-normalize`
ffmpeg-normalize $1/*.wav -of $2 -ext wav -ar 22050 -e="-ac 1"
