import os
import numpy as np


def make_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def load_api_dict(api_dict_path):
    api_dict = {"PAD": 0, "UNK": 1, "HOLE": 2}
    ids = 3
    with open(api_dict_path, 'r') as f:
        for line in f:
            api_dict[line.strip()] = ids
            ids += 1
    return api_dict


def load_class_dict(class_dict_path):
    class_dict = {"PAD": 0, "UNK": 1}
    ids = 2
    with open(class_dict_path, 'r') as f:
        for line in f:
            class_dict[line.strip()] = ids
            ids += 1
    return class_dict


def load_data_set(data_set_path):
    data = []
    with open(data_set_path, 'r') as f:
        for line in f:
            line = line.strip()
            sample = dict()
            items = line.split(";")
            sample['data'] = items[0]
            sample['label'] = items[1]
            data.append(sample)
    return data


def pad_seq(class_seq, api_seq, max_len):
    if len(api_seq) < max_len:
        api_seq.extend(['PAD'] * max_len)
        class_seq.extend(['PAD'] * max_len)
        api_seq = api_seq[:max_len]
        class_seq = class_seq[:max_len]
    else:
        hole_loc = api_seq.index('HOLE')
        begin_loc = int(hole_loc - max_len/2)
        if begin_loc < 0:
            api_seq = api_seq[:max_len]
            class_seq = class_seq[:max_len]
        else:
            end_loc = int(hole_loc + max_len/2)
            if end_loc > len(api_seq):
                api_seq = api_seq[-max_len:]
                class_seq = class_seq[-max_len:]
            else:
                api_seq = api_seq[begin_loc: end_loc]
                class_seq = class_seq[begin_loc: end_loc]
    return class_seq, api_seq
