import sys
sys.path.append('./src')
import os
import os.path
from math import log, sqrt, ceil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from mel_utils import summarize

 #TODO register start position in buffer
class SequentialData(nn.Module):

    # make new copy of sequence internals for multiple data loader workers
    def instance(self):
        raise NotImplementedError()

    # get sequence data
    def get(self, seqname, coord_start, coord_end):
        raise NotImplementedError()

    @property
    def seqnames(self):
        raise NotImplementedError()

    @property
    def starts(self):
        raise NotImplementedError()

    @property
    def ends(self):
        raise NotImplementedError()

    def all_intervals(self):
        table = {
            'seqname': self.seqnames,
            'start': self.starts,
            'end': self.ends,
        }
        # print(table)
        return table


class MelDir(SequentialData):

    def __init__(self, directory):
        self.dir = directory
        self.unshuffled_names = os.listdir(self.dir)
        self.names = os.listdir(self.dir)
        np.random.shuffle(self.names)

    def instance(self):
        pass

    def get(self, seqname, coord_start, coord_end):
        return torch.load(os.path.join(self.dir, seqname))[:, coord_start:coord_end]

    @property
    def seqnames(self):
        return self.names

    @property
    def starts(self):
        return [0] * len(self.names)

    @property
    def ends(self):
        return [torch.load(os.path.join(self.dir, name)).shape[1] for name in self.names]


class StridedSequence(IterableDataset):

    """
        sequential: if True, this is equivalent to setting stride=1 and start_offset=0
    """
    def __init__(self, sequence_data, seq_len, include_intervals=None,
                transforms=torch.nn.Identity(),
                sequential=False, stride=None, start_offset=None):
        self.sequence_data = sequence_data
        self.seq_len = seq_len
        self._cutoff = self.seq_len - 1
        self.transforms = transforms
        self.include_intervals = include_intervals
        if self.include_intervals is None:  # use entire sequence
            self.include_intervals = self.sequence_data.all_intervals()
        self._gen_interval_table(self.include_intervals)

        if sequential:  # return sequences in order from beginning
            self.stride = 1
            self.start_offset = 0
        else:
            if stride is None:
                # make sure total positions is odd (this guarantees stride covers all positions)
                if self.n_seq % 2 == 0:
                    self.n_seq -= 1
                # nearest power of 2 to square root of self.n_seq, this gives nicely spaced positions
                self.stride = 2 ** int(round(log(sqrt(self.n_seq), 2)) + 3) #TODO this is a hack to get it to move between files more
            else:
                self.stride = stride
            # randomly assign start position (this will be different for each dataloader worker)
            if start_offset is None:
                self.start_offset = torch.randint(self.n_seq, [1]).item()
            else:
                self.start_offset = start_offset

    def _gen_interval_table(self, include_intervals):
        # if length is negative, remove interval (set length to 0)
        lengths = [max(0, y - x - self._cutoff)
                    for x, y in zip(include_intervals['start'], include_intervals['end'])]
        self.keys = include_intervals['seqname']
        self.coord_offsets = list(include_intervals['start'])
        self.n_seq = np.sum(lengths)
        self.last_indexes = np.cumsum(lengths)
        self._iter_start = 0
        self._iter_end = self.n_seq

    @staticmethod
    def _worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.sequence_data.instance()  # need new object for concurrency
        n_per_worker = int(ceil(dataset.n_seq / worker_info.num_workers))
        dataset._iter_start = n_per_worker * worker_info.id  # split indexes by worker id
        dataset._iter_end = min(dataset._iter_start + n_per_worker, dataset.n_seq)

    def get_data_loader(self, batch_size, num_workers):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, worker_init_fn=self._worker_init_fn)

    def index_to_coord(self, i):
        index = (i * self.stride + self.start_offset) % self.n_seq
        # look for last (right side) matching value to skip over any removed (zero length) intervals
        row = np.searchsorted(self.last_indexes, index, side='right')
        # look up sequence name and genomic coordinate from interval table
        key = self.keys[row]
        if row == 0:  # need to find index relative to start of interval
            index_offset = 0
        else:
            index_offset = self.last_indexes[row - 1]
        coord =  self.coord_offsets[row] + index - index_offset
        return key, coord

    def __iter__(self):
        for i in range(self._iter_start, self._iter_end):
            key, coord = self.index_to_coord(i)
            seq = self.sequence_data.get(key, coord, coord + self.seq_len)
            # static = detect_static(seq)
            # if static is not None:
            #     print(key, coord, static)
            seq = self.transforms(seq)
            print('    ', key, coord, summarize(seq))
            yield seq, (key, coord)


def detect_static(mel_tensor):
    # detect static (all -4.5 across all 80 channels) using mean across channels
    is_static = torch.sum(mel_tensor, axis=0) == -4.5 * 80
    if torch.any(is_static):
        static_idx = torch.where(is_static)[0]
        non_consecutive = (static_idx[1:] - static_idx[:-1]) != 1
        return static_idx
        # indexes = torch.index_select(static_idx, 0, torch.where(non_consecutive)[0])
        # return indexes
    return None

