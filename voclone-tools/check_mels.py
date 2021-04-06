import os
import os.path
import argparse
import torch


def find_static_in_dir(source_dir):
    files = os.listdir(source_dir)
    for f in files:
        mel = torch.load(os.path.join(source_dir, f))
        mel = mel[:, :mel.shape[1] // 2]  # check only train half
        indexes = detect_static(torch.tensor(mel))
        if indexes is not None:
            print(f, indexes)

def detect_static(mel_tensor):
    # detect static (all -4.5 across all 80 channels) using mean across channels
    is_static = torch.sum(mel_tensor, axis=0) == -4.5 * 80
    if torch.any(is_static):
        static_idx = torch.where(is_static)[0]
        non_consecutive = (static_idx[1:] - static_idx[:-1]) != 1
        return static_idx
        indexes = torch.index_select(static_idx, 0, torch.where(non_consecutive)[0])
        return indexes
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    args = parser.parse_args()
    find_static_in_dir(args.i)
