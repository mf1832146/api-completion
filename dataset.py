from torch.utils.data import Dataset, DataLoader
import torch
import collections


class MyDataSet(Dataset):
    def __init__(self, data_path, api_dict, class_dict, class_to_api_dict, max_class_len, max_api_len):
        self.api_dict = api_dict
        self.class_dict = class_dict
        self.class_to_api_dict = class_to_api_dict
        self.max_api_len = max_api_len
        self.max_class_len = max_class_len
        self.max_candidate_api_len = 200
        self.data_set = load_data_set(data_path)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        sample = self.data_set[index]

        data = sample['data']
        class_seq, api_seq, hole_class = self.get_sample(data)
        for i in range(len(api_seq)):
            if len(api_seq[i]) < self.max_api_len:
                api_seq[i].extend(['PAD'] * self.max_api_len)
                api_seq[i] = api_seq[i][:self.max_api_len]
        if len(class_seq) < self.max_class_len:
            class_seq.extend(['PAD'] * self.max_class_len)
            class_seq = class_seq[:self.max_class_len]
            api_seq.extend([['PAD'] * self.max_api_len] * self.max_class_len)
            api_seq = api_seq[:self.max_class_len]

        class_seq = [self.class_dict[x] if x in self.class_dict else 1 for x in class_seq]
        api_seq = [[self.api_dict[x] if x in self.api_dict else 1 for x in y] for y in api_seq]

        candidate_api_seq = self.class_to_api_dict[self.class_dict[hole_class] if hole_class in self.class_dict else 1]
        if len(candidate_api_seq) < self.max_candidate_api_len:
            candidate_api_seq.extend([0] * self.max_candidate_api_len)
            candidate_api_seq = candidate_api_seq[:self.max_candidate_api_len]
        else:
            candidate_api_seq = candidate_api_seq[:self.max_candidate_api_len]
        label = sample['label']
        label = self.api_dict[label] if label in self.api_dict else 1

        if label not in candidate_api_seq:
            candidate_api_seq[0] = label
        label = candidate_api_seq.index(label)

        class_seq = torch.LongTensor(class_seq)
        api_seq = torch.LongTensor(api_seq)
        candidate_api_seq = torch.LongTensor(candidate_api_seq)

        return (class_seq, api_seq, candidate_api_seq), label

    def get_sample(self, api_seq):
        items = api_seq.split()
        hole_loc = items.index('HOLE')
        pre_class_seq, pre_api_seq = self.deal_with_sub_str(items[:hole_loc - 1], True)
        hole_class = items[hole_loc - 1]
        hole_api = 'HOLE'
        suffix_class_seq, suffix_api_seq = self.deal_with_sub_str(items[hole_loc + 1:], False)

        if len(pre_class_seq) + len(suffix_class_seq) > self.max_class_len - 1:
            if len(pre_class_seq) < self.max_class_len / 2:
                suffix_seq_len = self.max_class_len - len(pre_class_seq) - 1
                suffix_class_seq = suffix_class_seq[:suffix_seq_len]
                suffix_api_seq = suffix_api_seq[:suffix_seq_len]
            elif len(suffix_class_seq) < self.max_class_len / 2:
                pre_seq_len = self.max_class_len - len(suffix_class_seq) - 1
                pre_class_seq = pre_class_seq[-1 * pre_seq_len:]
                pre_api_seq = pre_api_seq[-1 * pre_seq_len:]
            else:
                pre_len = int(self.max_class_len / 2)
                suffix_len = self.max_class_len - 1 - pre_len
                suffix_class_seq = suffix_class_seq[:suffix_len]
                suffix_api_seq = suffix_api_seq[:suffix_len]
                pre_class_seq = pre_class_seq[-1 * pre_len:]
                pre_api_seq = pre_api_seq[-1 * pre_len:]

        pre_class_seq.append(hole_class)
        pre_class_seq.extend(suffix_class_seq)
        pre_api_seq.append([hole_api])
        pre_api_seq.extend(suffix_api_seq)
        return pre_class_seq, pre_api_seq, hole_class

    def deal_with_sub_str(self, sub_seq, is_pre):
        class_seq = []
        api_seq = []
        ordered_dict = collections.OrderedDict()
        for i in range(len(sub_seq) - 1, 0, -2):
            class_name = sub_seq[i - 1]
            api_name = class_name + '.' + sub_seq[i]
            if class_name not in ordered_dict:
                ordered_dict[class_name] = []
            ordered_dict[class_name].insert(0, api_name)

            if len(ordered_dict[class_name]) > self.max_api_len:
                seq = ordered_dict[class_name]
                if is_pre:
                    ordered_dict[class_name] = seq[-self.max_api_len:]
                else:
                    ordered_dict[class_name] = seq[:self.max_api_len]
        for key, item in ordered_dict.items():
            class_seq.insert(0, key)
            api_seq.insert(0, item)
        return class_seq, api_seq


class StandardDataSet(Dataset):
    def __init__(self, data_path, api_dict, class_dict, max_len):
        self.api_dict = api_dict
        self.class_dict = class_dict
        self.max_len = max_len
        self.data_set = load_data_set(data_path)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        """pad the seq for each sample"""
        sample = self.data_set[index]

        data = sample['data']
        api_seq = []
        class_seq = []
        items = data.split()
        for i, api in enumerate(items):
            if i % 2 == 0:
                class_seq.append(api)
            else:
                if api == "HOLE":
                    api_seq.append('HOLE')
                else:
                    api_seq.append(items[i - 1] + '.' + api)

        class_seq, api_seq = pad_seq(class_seq, api_seq, self.max_len)
        api_seq = [self.api_dict[x] if x in self.api_dict else 1 for x in api_seq]
        class_seq = [self.class_dict[x] if x in self.class_dict else 1 for x in class_seq]

        label = sample['label']
        label = self.api_dict[label] if label in self.api_dict else 1

        """list to tensor"""
        api_seq = torch.LongTensor(api_seq)
        class_seq = torch.LongTensor(class_seq)
        # label = torch.LongTensor(label)

        return (class_seq, api_seq), label


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


def get_data_loaders(api_dict, class_dict, class_to_api_dict, args):
    if args.model == 'lstm':
        train_data_set = StandardDataSet(args.data_dir + 'train/data_set.txt',
                                         api_dict,
                                         class_dict,
                                         args.api_max_len)

        valid_data_set = StandardDataSet(args.data_dir + 'valid/data_set.txt',
                                         api_dict,
                                         class_dict,
                                         args.api_max_len)
    elif args.model == 'my_model':
        train_data_set = MyDataSet(args.data_dir + 'train/data_set.txt',
                                   api_dict,
                                   class_dict,
                                   class_to_api_dict,
                                   args.class_max_len,
                                   args.api_max_len)
        valid_data_set = MyDataSet(args.data_dir + 'valid/data_set.txt',
                                   api_dict,
                                   class_dict,
                                   class_to_api_dict,
                                   args.class_max_len,
                                   args.api_max_len)
    train_loader = DataLoader(dataset=train_data_set,
                              batch_size=args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_data_set,
                              batch_size=args.batch_size,
                              shuffle=False)
    return train_loader, valid_loader
