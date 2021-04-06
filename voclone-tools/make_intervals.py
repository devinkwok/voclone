import os
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F


class SeqIntervals():

    _DEFAULT_COL_NAMES = ['names', 'start', 'end']

    """
    Defines start and endpoints within a set of named sequences.
    Intervals must be unique (by name) and non-overlapping.

    seqname_start_end_index: list of int indices or str names of columns relevant
            SeqIntervals, must be in order [seqname, start, end]
    """
    def __init__(self, annotation_table, seqname_start_end_index=[0, 1, 2], nonzero=True):
        self.table = annotation_table
        self._index = pd.Index([])
        for i, col in enumerate(seqname_start_end_index):
            if type(col) is int:
                self._index = self._index.insert(i, annotation_table.columns[col])
            else:
                assert col in annotation_table.columns
                self._index = self._index.insert(i, col)
        if nonzero:
            self.table = self._filter_by_len()  # get rid of 0 or negative length intervals

    @classmethod
    def from_cols(cls, seqs, starts, ends, nonzero=True):
        data =  {cls._DEFAULT_COL_NAMES[0]: seqs,
                cls._DEFAULT_COL_NAMES[1]: starts,
                cls._DEFAULT_COL_NAMES[2]: ends}
        return cls(pd.DataFrame(data=data, columns=cls._DEFAULT_COL_NAMES), nonzero=nonzero)

    @classmethod
    def from_bed_file(cls, filename, sep='\t', nonzero=True):
        with open(filename, 'r') as file:
            table = pd.read_csv(file, sep=sep, names=cls._DEFAULT_COL_NAMES)
        return cls(table, nonzero=nonzero)

    @property
    def seqnames(self):
        return self.table[self._index[0]]

    @property
    def start(self):
        return self.table.loc[:, self._index[1]]

    @property
    def end(self):
        return self.table.loc[:, self._index[2]]

    @property
    def length(self):
        return self.end - self.start
    
    def __len__(self):
        return len(self.table)

    def __repr__(self):
        return repr(self.table)

    def __str__(self):
        return repr(self.table)

    def clone(self, new_table=None, deep_copy=False):
        table = new_table
        if new_table is None:
            table = self.table
        if deep_copy:
            table = table.copy()
        return SeqIntervals(table, self._index, nonzero=False)

    def filter(self, *loc_labels, col_to_search=None):
        if col_to_search is None:
            col_to_search = self.seqnames
        return self.clone(self.table[col_to_search.isin(loc_labels)], False)

    def _filter_by_len(self, min_len=1, max_len=None):
        if max_len is None:
            return self.table.loc[self.length >= min_len]
        else:  # use element-wise and `&`
            return self.table.loc[(self.length >= min_len) & (self.length <= max_len)]

    # max_len is inclusive
    def filter_by_len(self, min_len=1, max_len=None):
        return self.clone(self._filter_by_len(min_len, max_len), False)

    # reimplement this for labelled intervals
    def columns_match(self, interval):
        return (self.table.columns == interval.table.columns).all() \
                and (self._index == interval._index).all()

    """
    Set union (addition) of self SeqIntervals object with other SeqIntervals

        in_place: default `False` creates a new copy of underlying table data.
            If `in_place=True`, modify calling object's table.
    """
    def union(self, *intervals, min_allowable_gap=1, in_place=False):
        table = self._append_tables(*intervals)
        if in_place:
            self.table = table
            interval_obj = self
        else:
            interval_obj = self.clone(table, True)
        return interval_obj._merge_overlaps(*interval_obj._find_overlaps(
                                    min_allowable_gap), min_allowable_gap)

    """
    Set intersection (product). If run on self with no other SeqIntervals objects,
    this will find all overlapping intervals in this object, including nested overlaps.
    In particular, if A is contained in B and B is contained in C,
    it will return both B and A.
    
    If run on at least one other SeqIntervals object, this will merge within-object overlaps
    and find between-object overlaps common to all SeqIntervals. That is, it makes each object
    non-self-intersecting and finds the set intersection of all objects.

        in_place: if `True`, all objects in `*intervals` will have union() run on them
            to remove overlaps and redundant intervals
    """
    def intersect(self, *intervals, in_place=False):
        if len(intervals) == 0:
            first = self
            if not in_place:
                first = self.clone()
            return first._isolate_overlaps(*first._find_overlaps(1))
            # raise ValueError('Need at least one other SeqIntervals obj to intersect with.')
        first = self.union(in_place=in_place)
        for interval in intervals:
            interval = interval.union(in_place=in_place)  # get rid of overlaps
            first.table = first._append_tables(interval)
            first = first._isolate_overlaps(*first._find_overlaps(1))
        return first

    def remove(self, *intervals, in_place=False):
        if len(intervals) == 0:
            raise ValueError('No intervals to remove.')
        remove_intervals = intervals[0].union(*intervals[1:], in_place=False)
        remove_intervals = self.intersect(remove_intervals, in_place=False)

    def _append_tables(self, *intervals):
        new_table = self.table
        for i in intervals:
            if self.columns_match(i):
                new_table = new_table.append(i.table)
            else:
                raise ValueError('SeqInterval object columns do not match ', str(interval))
        # need to re-index with reset_index to avoid duplicate indices, and don't keep the old index
        return new_table.reset_index(drop=True)

    # warning: this is an in place operation!
    def _find_overlaps(self, min_allowable_gap):
        self.table = self.table.sort_values([self._index[0], self._index[1]])
        distance_to_next_interval = self.start[1:].values - self.end[:-1].values
        is_same_sequence = self.seqnames[1:].values == self.seqnames[:-1].values
        # overlap_candidates guaranteed to contain the first of multiple overlapping positions
        # but not necessarily all of the subsequent positions: counter example is
        # 0-10, 1-2, 3-4, overlap_candidates will be [True, False] even though 3-4 overlaps 0-10
        # this gives a starting point for iterating over intervals sequentially, which is slow
        overlap_candidates = (distance_to_next_interval < min_allowable_gap) & is_same_sequence
        return is_same_sequence, overlap_candidates

    # warning: this is an in place operation!
    def _merge_overlaps(self, is_same_sequence, overlap_candidates, min_allowable_gap):
        rows_to_remove = []
        end_col_index = self.table.columns.get_loc(self._index[2])
        for i in np.nonzero(overlap_candidates)[0]:
            j = i + 1
            # use this to avoid iterating over same positions in both while and for loop
            if overlap_candidates[i]:
                # check if overlap_candidates are between the same named sequence
                while j < len(self.table) and is_same_sequence[j - 1] \
                        and self.start.iloc[j] - self.end.iloc[i] < min_allowable_gap:
                    # if true, merge the intervals and the next ones
                    # until the distance to next is greater than min_allowable_gap
                    # extend the interval at position i
                    # note: cannot update array values using chained indexing, hence use `.iloc`
                    # see https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
                    self.table.iloc[i, end_col_index] = max(self.end.iloc[i], self.end.iloc[j])
                    rows_to_remove.append(j)  # remove subsequent intervals
                    overlap_candidates[j - 1] = False  # set this to avoid iterating again in for loop
                    j += 1
        self.table = self.table.drop(self.table.iloc[rows_to_remove].index)
        return self

    # warning: this is an in place operation!
    # assume that there are no more than 2 overlapping intervals at any point in the sequence
    # this is accomplished by combining the tables of two SeqIntervals that have run union()
    # to merge overlaps
    def _isolate_overlaps(self, is_same_sequence, overlap_candidates):
        start_col_index = self.table.columns.get_loc(self._index[1])
        end_col_index = self.table.columns.get_loc(self._index[2])
        rows_to_keep = []
        for i in np.nonzero(overlap_candidates)[0]:
            j = i + 1
            start_coord, end_coord = self.start.iloc[i], self.end.iloc[i]
            while j < len(self.table) and is_same_sequence[j - 1] \
                    and self.start.iloc[j] - end_coord < 0:
                # set position j to the intersection coordinates, adding padding
                self.table.iloc[j - 1, start_col_index] = max(start_coord, self.start.iloc[j])
                self.table.iloc[j - 1, end_col_index] = min(end_coord, self.end.iloc[j])
                rows_to_keep.append(j - 1)
                # if the next interval is not contained in the current one it was found
                # by _find_overlaps, since there are no more than 2 overlapping intervals
                if self.end.iloc[j] >= end_coord:
                    break  # the for loop will iterate over the next overlap
                j += 1
        self.table = self.table.iloc[rows_to_keep]
        return self

    # warning: this is an in place operation!
    def _remove_overlaps(self, is_same_sequence, overlap_candidates, min_allowable_gap):
        rows_to_remove = []
        end_col_index = self.table.columns.get_loc(self._index[2])
        for i in np.nonzero(overlap_candidates)[0]:
            j = i + 1
            # use this to avoid iterating over same positions in both while and for loop
            if overlap_candidates[i]:
                # check if overlap_candidates are between the same named sequence
                while j < len(self.table) and is_same_sequence[j - 1] \
                        and self.start.iloc[j] - self.end.iloc[i] < min_allowable_gap:
                    # if true, merge the intervals and the next ones
                    # until the distance to next is greater than min_allowable_gap
                    # extend the interval at position i
                    # note: cannot update array values using chained indexing, hence use `.iloc`
                    # see https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
                    self.table.iloc[i, end_col_index] = max(self.end.iloc[i], self.end.iloc[j])
                    rows_to_remove.append(j)  # remove subsequent intervals
                    overlap_candidates[j - 1] = False  # set this to avoid iterating again in for loop
                    j += 1
        self.table = self.table.drop(self.table.iloc[rows_to_remove].index)
        return self


COL_NAMES = ['names', 'start', 'end']


def load_wav(filename):
    return sf.read(filename)[0]

# def moving_average(tensor, width, stride=1, padding=0):
#     channels = tensor.shape[0]
#     div_factor = width * channels
#     out = F.conv1d(tensor.unsqueeze(0), torch.ones(1, channels, width, dtype=torch.float) / div_factor,
#                     stride=stride, padding=padding)
#     return out.squeeze()

# def label_silence(mel):
#     values = moving_average(mel, 1000)
#     THRESHOLD = -10.
#     return (values + THRESHOLD) < 0

def names_from_dir(dirname):
    target = []
    for name in sorted(os.listdir(dirname)):
        split = name.split('-')
        prefix = '-'.join(split[:-1])
        suffix = split[-1]
        if not ('voice' in suffix) and not ('mask' in suffix):
            target.append(name.split('.')[0])
    return target

def target_intervals(dirname, basename):
    target_len = len(load_wav(os.path.join(dirname, basename + '.wav')))
    return [basename, 0, target_len]

def ma_intervals(dirname, basename, ma_window, threshold, file_suffix):
    print(basename)
    intervals = []
    midpoint_offset = int(ma_window // 2)
    try:
        sequence = load_wav(os.path.join(dirname, basename) + file_suffix)
    except:
        return intervals
    running_sum = sum([x * x for x in sequence[:ma_window]])
    is_gt_threshold = False
    interval_start = 0
    for i, (prev_s, next_s) in enumerate(zip(sequence[: - ma_window], sequence[ma_window:])):
        if i % 22050 == 0:
            print(running_sum / ma_window)
        if not is_gt_threshold and running_sum / ma_window > threshold:
            interval_start = i + midpoint_offset
            is_gt_threshold = True
        if is_gt_threshold and running_sum / ma_window < threshold:
            intervals.append([basename, interval_start, i + midpoint_offset])
            print('interval', [basename, interval_start, i + midpoint_offset])
            is_gt_threshold = False
        running_sum += next_s * next_s - prev_s * prev_s
    if is_gt_threshold:
        intervals.append([basename, interval_start, len(sequence)])
    return intervals

def open_intervals(filename):
    return pd.read_csv(filename, sep='\t', names=COL_NAMES)

def write_intervals(array, filename):
    intervals = pd.DataFrame(data=array, columns=COL_NAMES)
    intervals.to_csv(filename, sep='\t', index=False, header=False)

def gen_intervals(args):
    basenames = names_from_dir(args.input_dir)
    print(basenames)

    write_intervals([target_intervals(args.input_dir, name) for name in basenames],
            os.path.join(args.output_dir, 'intervals-target.csv'))

    source_intervals = []
    for name in basenames:
        [source_intervals.append(row) for row in ma_intervals(
                args.input_dir, name, args.window, args.threshold, '-voice.wav')]
    write_intervals(source_intervals, os.path.join(args.output_dir, 'intervals-source.csv'))

    mask_intervals = []
    for name in basenames:
        [mask_intervals.append(row) for row in ma_intervals(
            args.input_dir, name, args.window, args.threshold, '-mask.wav')]
    write_intervals(mask_intervals, os.path.join(args.output_dir, 'intervals-mask.csv'))

def write_wav(filename, seq, min_len):
    if len(seq) >= min_len:
        sf.write(filename + '.wav', seq, 22050)

def split_audio_using_intervals(dirname, basename, suffix, intervals, pos_dir, neg_dir, min_len):
    sequence = load_wav(os.path.join(dirname, basename + suffix))
    cur_pos = 0
    n = 0
    for _, (name, start, end) in intervals.iterrows():
        if name == basename:
            if cur_pos < start:
                write_wav(os.path.join(neg_dir, basename + str(n)), sequence[cur_pos:start], min_len)
                n += 1
            write_wav(os.path.join(pos_dir, basename + str(n)), sequence[start:end], min_len)
            cur_pos = end
            n += 1
    if cur_pos < len(sequence):
        write_wav(os.path.join(neg_dir, basename + str(n)), sequence[cur_pos:], min_len)

def make_dirs(*subdirs):
    for dirname in subdirs:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

def split_by_source(dirname, output_dir, min_len):
    pos_dir = os.path.join(output_dir, 'target')
    neg_dir = os.path.join(output_dir, 'noise')
    pos_source_dir = os.path.join(output_dir, 'source')
    neg_source_dir = os.path.join(output_dir, 'silence')
    pos_masked_dir = os.path.join(output_dir, 'mask_and_target')
    neg_masked_dir = os.path.join(output_dir, 'noise_no_mask')
    make_dirs(pos_dir, neg_dir, pos_source_dir, neg_source_dir, pos_masked_dir, neg_masked_dir)
    names = names_from_dir(dirname)
    intervals = open_intervals(os.path.join(output_dir, 'intervals-source.csv'))
    mask_intervals = open_intervals(os.path.join(output_dir, 'intervals-mask.csv'))
    mask_intervals = SeqIntervals(mask_intervals).union(SeqIntervals(intervals)).table
    for name in names:
        split_audio_using_intervals(dirname, name, '.wav', intervals, pos_dir, neg_dir, min_len)
        if os.path.exists(os.path.join(dirname, name + '-voice.wav')):
            split_audio_using_intervals(dirname, name, '-voice.wav', intervals, pos_source_dir, neg_source_dir, min_len)
        split_audio_using_intervals(dirname, name, '.wav', mask_intervals, pos_masked_dir, neg_masked_dir, min_len)

def classify_by_intervals(input_csv, threshold, output_csv, ch_rate=80, sample_rate=22050, n_running_mean=4, min_interval_ch_len=160):
    classification = pd.read_csv(input_csv, sep=',')
    intervals = []
    is_gt_threshold = False
    interval_start = 0
    prev_name = ''
    prev_position = 0
    running_val = [0] * n_running_mean
    running_idx = 0
    for _, row in classification.iterrows():
        _, name, position, logit = row
        running_val[running_idx] = logit
        test_val = sum(running_val) / n_running_mean
        running_idx = (running_idx + 1) % n_running_mean
        if prev_name != name and is_gt_threshold:  # write last interval if goes to end of seq
            if (prev_position - interval_start) >= min_interval_ch_len:
                intervals.append([prev_name, int(interval_start * sample_rate / ch_rate),
                                int((prev_position) * sample_rate / ch_rate)])
            is_gt_threshold = False
        if (not is_gt_threshold) and (test_val > threshold):  # start sequence
            interval_start = position
            is_gt_threshold = True
        if is_gt_threshold and (test_val < threshold):
            if (prev_position - interval_start) >= min_interval_ch_len:
                intervals.append([prev_name, int(interval_start * sample_rate / ch_rate),
                            int(position * sample_rate / ch_rate)])
            is_gt_threshold = False
        prev_name = name
        prev_position = position
    write_intervals(intervals, output_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--operation', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.0001)
    parser.add_argument('--window', type=int, default=22050)
    parser.add_argument('--min_len', type=int, default=22050)
    args = parser.parse_args()
    print(args)
    if args.operation == 'generate':
        gen_intervals(args)
    elif args.operation == 'split':
        split_by_source(args.input_dir, args.output_dir, args.min_len)
    elif args.operation == 'classify':
        classify_by_intervals(args.input_dir, args.threshold, args.output_dir)
