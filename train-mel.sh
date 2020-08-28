#!/bin/bash

## for testing 
python main.py \
    --mel_spectrogram=True \
    --dataset=meltest \
    --decay_flag=False \
    --light=True \
    --iteration=1005 \
    --print_freq=1 \
    --save_freq=1 \
    --ch=64 \
    --n_res=4 \
    --n_dis=4 \
    --img_size=80 \
    --img_ch=1 \
    --resume=True \
    --print_wandg=True \
    --print_input=True \
    --deterministic=True \
    --DEBUG_use_old=True \
    # --use_noise=True \
    # --gen_noise_A=0.5 \
    # --dis_noise_B=0.5 \
# python main.py \
#     --mel_spectrogram=True \
#     --light=True \
#     --dataset=ravdess_mel_gendersplit_normalized \
#     --iteration=1000000 \
#     --print_freq=500 \
#     --save_freq=500 \
#     --ch=64 \
#     --n_res=4 \
#     --n_dis=4 \
#     --img_size=80 \
#     --img_ch=1 \
#     --print_wandg=True \
#     --print_input=True \
#     --resume=True \
#     --use_noise=True \
#     --gen_noise_A=0.5 \
#     --dis_noise_B=0.5 \
