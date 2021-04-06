import argparse
import os.path
import cv2
import torch
import torch.nn.functional as F

from mel_utils import print_and_summarize, mel_to_img, mel_transform, inv_mel_transform, add_mels
from synthwaveglow import load_waveglow, mel2wav, save_soundfile


def pad(mel1, mel2, length, random_offset=True):
    len1 = mel1.shape[2]
    len2 = mel2.shape[2]
    max_len = max(len1, len2)
    mel1 = F.pad(mel1, pad=(0, max_len - len1), mode='constant', value=0)
    mel2 = F.pad(mel2, pad=(0, max_len - len2), mode='constant', value=0)
    target_len = min(max_len, length)
    if random_offset:
        offset = torch.randint(max_len - length, [1]).item()
    else:
        offset = 0
    return mel1[:, :, offset:offset+length], mel2[:, :, offset:offset+length]


def add(mel1, mel2):
    return mel_transform(torch.log(torch.exp(inv_mel_transform(mel1)) \
                    + torch.exp(inv_mel_transform(mel2))))


# adds log-mel spectrograms together at given proportion (transforms first by exp)
def add_noise(file1, file2, out_dir, length, chunk_size):
    source = mel_transform(torch.load(file1))
    noise = mel_transform(torch.load(file2))
    print_and_summarize(source, noise)

    source, noise = pad(source, noise, length)
    # source_prop = 1. - noise_prop
    print_and_summarize(source, noise)
    out = add_mels(source, noise)
    print_and_summarize(out)
    mel = torch.cat([source.detach().cpu(), noise.detach().cpu(), out.detach().cpu()],
                axis=2)
    torch.save(inv_mel_transform(mel), os.path.join(out_dir, 'combined.mel'))
    print_and_summarize(mel)
    cv2.imwrite(os.path.join(out_dir, 'combined.png'), mel_to_img(mel.squeeze()))
    
    audio = mel2wav(load_waveglow(), inv_mel_transform(mel), chunk_size)
    save_soundfile(audio, os.path.join(out_dir, 'combined.wav'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-j', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('-l', type=int, default=1000)
    parser.add_argument('--chunk_size', type=int, default=80)
    args = parser.parse_args()
    add_noise(args.i, args.j, args.o, args.l, args.chunk_size)
