

python main.py \
    --phase=NOISE_test \
    --mel_spectrogram=True \
    --light=False \
    --dataset=cohenfinal \
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
    --gen_noise_B=1.0 \
    --adv_weight=1 \
    --cycle_weight=10 \
    --identity_weight=10 \
    --cam_weight=1000 \
    --resume=True \
    --NOISE_n_samples 100 \
    --adjust_noise_volume=False \
    --scale_volume_A=0. \
    --scale_volume_B=0. \
    --scale_volume_noise=0. \
    --noise_margin=0. \

# python main.py \
#     --mel_spectrogram=True \
#     --phase=test \
#     --light=False \
#     --dataset=cohenfinal \
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
#     --adv_weight=1 \
#     --cycle_weight=10 \
#     --identity_weight=10 \
#     --cam_weight=1000 \
#     --resume=True \
#     --test_stride=160 \
#     --test_interpolate=False \

# python main.py \
#     --mel_spectrogram=True \
#     --phase=test \
#     --light=True \
#     --dataset=ravdess_mel_gendersplit_normalized \
#     --iteration=400000 \
#     --print_freq=5000 \
#     --save_freq=5000 \
#     --ch=64 \
#     --n_res=4 \
#     --n_dis=4 \
#     --img_size=160 \
#     --img_ch=1 \
#     --print_wandg=True \
#     --print_input=True \
#     --resume=True \
#     --test_stride=80 \
#     --test_interpolate=True \
#     # --use_noise=True \
#     # --gen_noise_A=0.5 \
#     # --dis_noise_B=0.5 \
