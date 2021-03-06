#!/bin/bash

# ## this is the big final model
python main.py \
    --mel_spectrogram=True \
    --num_workers=1 \
    --light=False \
    --dataset=cohen160-with-interview \
    --iteration=600000 \
    --print_freq=1000 \
    --save_freq=1000 \
    --ch=64 \
    --n_res=4 \
    --n_dis=6 \
    --img_size=160 \
    --mel_channels=160 \
    --img_ch=1 \
    --print_wandg=False \
    --print_input=True \
    --use_noise=True \
    --gen_noise_A=0. \
    --gen_noise_B=1. \
    --dis_noise_A=0. \
    --dis_noise_B=0. \
    --dis_noise_A2B=0. \
    --dis_noise_B2A=0. \
    --adv_weight=1 \
    --cycle_weight=1 \
    --identity_weight=10 \
    --cam_weight=1000 \
    --resume=True \
    --adjust_noise_volume=True \
    --scale_source_volume_A=0. \
    --scale_source_volume_B=0. \
    --scale_target_volume_A=0.08 \
    --scale_target_volume_B=0.25 \
    --scale_source_volume_noise=0.1 \
    --scale_target_volume_noise=0.25 \
    --noise_margin=0.02 \
    --noise_weight_A=10. \
    --noise_weight_B=0. \
    --w_sin_freq_A=1.5 \
    --h_sin_freq_A=3. \
    --w_sin_amp_A=0.06 \
    --h_sin_amp_A=0.01 \
    --sp_amp_A=0.005 \
    --w_stretch_max_A=40 \
    --h_translate_max_A=0 \
    --w_sin_freq_B=1.5 \
    --h_sin_freq_B=3. \
    --w_sin_amp_B=0.1 \
    --h_sin_amp_B=0.02 \
    --sp_amp_B=0.01 \
    --w_stretch_max_B=60 \
    --h_translate_max_B=0 \
    --w_sin_freq_noise=1.5 \
    --h_sin_freq_noise=3. \
    --w_sin_amp_noise=0.06 \
    --h_sin_amp_noise=0.02 \
    --sp_amp_noise=0.01 \
    --w_stretch_max_noise=40 \
    --h_translate_max_noise=0 \


# ## this is the small test model
# python main.py \
#     --phase=dis \
#     --test_stride=40 \
#     --mel_spectrogram=True \
#     --light=True \
#     --dataset=cohen \
#     --iteration=1000000 \
#     --print_freq=5000 \
#     --save_freq=5000 \
#     --ch=64 \
#     --n_res=4 \
#     --n_dis=4 \
#     --img_size=80 \
#     --mel_channels=80 \
#     --img_ch=1 \
#     --print_wandg=True \
#     --print_input=True \
#     --use_noise=True \
#     --gen_noise_A=0. \
#     --gen_noise_B=1. \
#     --adv_weight=2 \
#     --cycle_weight=10 \
#     --identity_weight=10 \
#     --cam_weight=1000 \
#     --resume=True \
#     --adjust_noise_volume=True \
#     --scale_source_volume_A=0. \
#     --scale_source_volume_B=0. \
#     --scale_target_volume_A=0.15 \
#     --scale_target_volume_B=0.15 \
#     --scale_source_volume_noise=0.1 \
#     --scale_target_volume_noise=0.25 \
#     --noise_margin=-0.1 \
#     --noise_weight_A=10. \
#     --noise_weight_B=0. \
#     --w_sin_freq=1.5 \
#     --h_sin_freq=3. \
#     --w_sin_amp=0.12 \
#     --h_sin_amp=0.04 \
#     --sp_amp=0.04 \
#     --w_stretch_max=60 \
#     --h_translate_max=1 \

# this version uses noise discriminator plus added noise
# python main.py \
#     --mel_spectrogram=True \
#     --light=False \
#     --dataset=cohendis \
#     --iteration=1000000 \
#     --print_freq=5000 \
#     --save_freq=5000 \
#     --ch=64 \
#     --n_res=4 \
#     --n_dis=6 \
#     --img_size=160 \
#     --mel_channels=80 \
#     --img_ch=1 \
#     --print_wandg=False \
#     --print_input=True \
#     --use_noise=True \
#     --gen_noise_A=0. \
#     --gen_noise_B=1. \
#     --dis_noise_A=0. \
#     --dis_noise_B=0. \
#     --dis_noise_B2A=0. \
#     --dis_noise_A2B=0. \
#     --adv_weight=2 \
#     --cycle_weight=10 \
#     --identity_weight=10 \
#     --cam_weight=1000 \
#     --resume=False \
#     --adjust_noise_volume=True \
#     --scale_volume_A=0. \
#     --scale_volume_B=0.25 \
#     --scale_volume_noise=0.25 \
#     --noise_margin=0.02 \
#     --noise_weight_A=1. \
#     --noise_weight_B=0. \

# this version: do not add any noise except to B discriminator on real B
# that way, A2B2A has noise, B2A2B doesn't have noise
# make A discriminator distinguish between noise and real A
# that way, B2A is pushed towards real A and not noise
# even though no noise is added to B2A
# python main.py \
#     --mel_spectrogram=True \
#     --light=True \
#     --dataset=cohen \
#     --iteration=1000000 \
#     --print_freq=5000 \
#     --save_freq=5000 \
#     --ch=64 \
#     --n_res=4 \
#     --n_dis=4 \
#     --img_size=80 \
#     --mel_channels=80 \
#     --img_ch=1 \
#     --print_wandg=False \
#     --print_input=True \
#     --use_noise=True \
#     --gen_noise_A=0. \
#     --gen_noise_B=0. \
#     --dis_noise_A=0. \
#     --dis_noise_B=1. \
#     --dis_noise_B2A=1. \
#     --dis_noise_A2B=0. \
#     --adv_weight=2 \
#     --cycle_weight=10 \
#     --identity_weight=10 \
#     --cam_weight=1000 \
#     --resume=True \
#     --adjust_noise_volume=True \
#     --scale_volume_A=0. \
#     --scale_volume_B=0.25 \
#     --scale_volume_noise=0.25 \
#     --noise_margin=0.02 \
#     --noise_weight_A=1. \
#     --noise_weight_B=0. \

# python main.py \
#     --mel_spectrogram=True \
#     --light=False \
#     --dataset=cohensmall \
#     --iteration=1000000 \
#     --print_freq=5000 \
#     --save_freq=5000 \
#     --ch=64 \
#     --n_res=4 \
#     --n_dis=6 \
#     --img_size=160 \
#     --mel_channels=80 \
#     --img_ch=1 \
#     --print_wandg=True \
#     --print_input=True \
#     --use_noise=True \
#     --gen_noise_B=1.0 \
#     --adv_weight=2 \
#     --cycle_weight=10 \
#     --identity_weight=10 \
#     --cam_weight=1000 \
#     --resume=True \
#     --adjust_noise_volume=True \
#     --scale_volume_A=0. \
#     --scale_volume_B=0.25 \
#     --scale_volume_noise=0.25 \
#     --noise_margin=0.02 \


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
