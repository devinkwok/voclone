import os
import os.path
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from make_intervals import open_intervals, write_intervals
from mel_utils import random_crop


def make_model(state_dict=None, kernel=1):
    model = nn.Sequential(
        nn.Conv1d(80, 1, kernel),
        # nn.ReLU(),
        # nn.Conv1d(20, 1, kernel),
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model

class MelDataset(torch.utils.data.Dataset):

    def __init__(self, pos_dir, neg_dir, target_len):
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        self.pos = os.listdir(self.pos_dir)
        self.neg = os.listdir(self.neg_dir)
        self.target_len = target_len

    def __len__(self):
        return len(self.pos) + len(self.neg)

    def __getitem__(self, index):
        if index < len(self.pos):
            path = os.path.join(self.pos_dir, self.pos[index])
            # mel = torch.ones(80, 120)
            value = 1.
        else:
            path = os.path.join(self.neg_dir, self.neg[index - len(self.pos)])
            # mel = torch.zeros(80, 120)
            value = 0.
        mel = torch.load(path).unsqueeze(0)
        mel = random_crop(mel, self.target_len, std_epsilon=-1., mean_epsilon=-100.)
        return mel.squeeze(), value

def train(model, dataset, batch_size, epochs, learn_rate, use_cuda, out_path):
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, num_workers=2, drop_last=True)

    model.to(device)
    for i in range(epochs):
        model.train()
        for j, (x, target) in enumerate(dataloader):
            x = x.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            y = F.adaptive_avg_pool1d(model.forward(x), 1).squeeze()
            # print(x[:,0,0], y, target)
            loss = loss_fn(y, target.to(device))
            acc = torch.sum(((y > 0).to(torch.float) - target) == 0).item() / y.nelement()
            print('epoch {} batch {} loss {} acc {}'.format(i, j, loss, acc))
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), os.path.join(out_path, 'voice-detector_{}.pth'.format(epochs)))

def infer(model, mel_dir, threshold):
    intervals = []
    for path in os.listdir(mel_dir):
        name = path.split('.')[0]
        mel = torch.load(os.path.join(mel_dir, path))
        labels = model.forward(mel.unsqueeze(0))
        labels = (labels > threshold).to(torch.float).squeeze()
        starts, ends = interval_extents(labels)
        [intervals.append([name, int(start), int(end)])
                for start, end in zip(starts, ends)]
    write_intervals(intervals, os.path.join(mel_dir, 'intervals-source.csv'))

def interval_extents(sequence, mel_rate=80, sample_rate=22050):
    seq = F.pad(sequence, pad=(1, 1), mode='constant', value=0)
    diff = torch.sub(seq[1:], seq[:-1])
    starts = np.floor(np.multiply(np.where(diff > 0)[0], sample_rate / mel_rate))
    ends = np.floor(np.multiply(np.where(diff < 0)[0], sample_rate / mel_rate))
    if ends[-1] > len(sequence) * sample_rate / mel_rate:
        ends[-1] = int(len(sequence) * sample_rate / mel_rate)
    return starts, ends

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_dir', type=str, required=True)
    parser.add_argument('--neg_dir', type=str, required=True)
    parser.add_argument('--kernel', type=int, default=1)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--data_frame_len', type=int, default=160)
    parser.add_argument('--learn_rate', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default='.')
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.)
    args = parser.parse_args()
    print(args)

    state_dict = None
    if args.model_path is not None and args.model_path != '':
        state_dict = torch.load(args.model_path)
    model = make_model(state_dict, args.kernel)
    data = MelDataset(args.pos_dir, args.neg_dir, args.data_frame_len)
    if args.train:
        train(model, data, args.batch, args.epochs, args.learn_rate, args.cuda, args.out_path)
    else:
        infer(model, args.pos_dir, args.threshold)

if __name__ == '__main__':
    main()
