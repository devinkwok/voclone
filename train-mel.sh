python main.py \
    --mel_spectrogram=True \
    --light=True \
    --dataset=ravdess_mel_gendersplit_normalized \
    --iteration=400000 \
    --print_freq=5000 \
    --save_freq=5000 \
    --ch=64 \
    --n_res=4 \
    --n_dis=4 \
    --img_size=80 \
    --img_ch=1 \
    --print_wandg=True \
    --print_input=True \
    --resume=True \
    # --use_noise=True \
    # --gen_noise_A=0.5 \
    # --dis_noise_B=0.5 \
