#!/bin/bash
for f in $(ls results/cohen160/model | sort -r);
do
# ## this is the big final model
python main.py \
    --phase=test \
    --mel_spectrogram=True \
    --num_workers=1 \
    --light=False \
    --dataset=cohen160 \
    --iteration=600000 \
    --print_freq=5000 \
    --save_freq=5000 \
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
    --gen_noise_B=0. \
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
    --noise_weight_A=1000. \
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

mv ./results/cohen160/test/fake_B2A_42.mel ./results/cohen160/done_test/fake_B2A_42_$f
mv ./results/cohen160/model/$f ./results/cohen160/done_models/$f
done