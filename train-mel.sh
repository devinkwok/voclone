#!/bin/bash

python main.py \
    --mel_spectrogram=True \
    --light=False \
    --dataset=cohensmall \
    --iteration=1000000 \
    --print_freq=5000 \
    --save_freq=5000 \
    --ch=64 \
    --n_res=4 \
    --n_dis=6 \
    --img_size=160 \
    --mel_channels=80 \
    --img_ch=1 \
    --print_wandg=True \
    --print_input=True \
    --use_noise=True \
    --gen_noise_B=11.0 \
    --adv_weight=2 \
    --cycle_weight=10 \
    --identity_weight=10 \
    --cam_weight=1000 \
    --resume=True \
    --adjust_noise_volume=True \
    --scale_volume_A=0. \
    --scale_volume_B=0.25 \
    --scale_volume_noise=0.25 \
    --noise_margin=0.02 \


# python main.py \
#     --mel_spectrogram=True \
#     --light=True \
#     --dataset=ravdess_mel_gendersplit_normalized \
#     --iteration=1000000 \
#     --print_freq=10000 \
#     --save_freq=10000 \
#     --ch=64 \
#     --n_res=4 \
#     --n_dis=4 \
#     --img_size=80 \
#     --img_ch=1 \
#     --print_wandg=True \
#     --print_input=True \
#     --resume=False \
#     --use_noise=True \
#     --gen_noise_A=1.0 \
#     --gen_noise_B=1.0 \
#     --identity_noise_A=0.0 \
#     --identity_noise_B=0.0 \
#     --cycle_noise_A=0.0 \
#     --cycle_noise_B=0.0 \
#     --adv_weight=1 \
#     --cycle_weight=10 \
#     --identity_weight=10 \
#     --cam_weight=1000 \

## for testing 
# python main.py \
#     --mel_spectrogram=True \
#     --dataset=meltest \
#     --decay_flag=False \
#     --light=True \
#     --iteration=1005 \
#     --print_freq=1 \
#     --save_freq=1 \
#     --ch=64 \
#     --n_res=4 \
#     --n_dis=4 \
#     --img_size=80 \
#     --img_ch=1 \
#     --resume=True \
#     --print_wandg=True \
#     --print_input=True \
#     --deterministic=True \
#     # --use_noise=True \
#     # --gen_noise_A=0.5 \
#     # --dis_noise_B=0.5 \
