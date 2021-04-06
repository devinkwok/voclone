import argparse
import torch
import os
import os.path
import cv2

from mel_utils import mel_transform, inv_mel_transform, print_and_summarize


def mels_to_imgs(input_dir, output_dir):
    names = os.listdir(input_dir)

    mels = [torch.tensor(torch.load(os.path.join(input_dir, n))) for n in names]

    melmean = torch.tensor([torch.std_mean(m)[1].item() for m in mels])
    melstd = torch.tensor([torch.std_mean(m)[0].item() for m in mels])
    melmax = torch.tensor([torch.max(m).item() for m in mels])
    melmin = torch.tensor([torch.min(m).item() for m in mels])
    print_and_summarize(melmin, melmean, melstd, melmax)

    imgs = [mel_to_img(normalize(m)) for m in mels]
    [cv2.imwrite(os.path.join(output_dir, os.path.splitext(n)[0] + '.png'), i)
                for n, i in zip(names, imgs)]


def normalize(mel, zero_intercept=-11, scale=13.):
    return (mel - zero_intercept) / scale


def mel_to_img(mel):
    red = torch.clamp(mel * 3 * 256, 0, 256)
    green = torch.clamp(mel * 3 * 256, 256, 512) - 256
    blue = torch.clamp(mel * 3 * 256, 512, 768) - 512
    img = torch.stack([blue, green, red], dim=0)
    return torch.flip(img, [1]).permute(1,2,0).numpy()


def labels_to_img(*labels):
    for label in labels:  # assume they are all boolean
        pass #TODO


def labelled_mel_to_img(mel, *labels):
    img = mel_to_img(mel)
    img_label = labels_to_img(*labels)
    return torch.cat([img, img_label], dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    mels_to_imgs(args.input_dir, args.output_dir)
