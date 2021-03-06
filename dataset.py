from torch.utils.data import Dataset, DataLoader
import torch
import collections


class MyDataSet(Dataset):
    def __init__(self, data_path, api_dict, class_dict, class_to_api_dict, max_class_len, max_api_len, need_unique=False):
        self.api_dict = api_dict
        self.class_dict = class_dict
        self.class_to_api_dict = class_to_api_dict
        self.max_api_len = max_api_len
        self.max_class_len = max_class_len
        self.max_candidate_api_len = 200
        self.data_set = load_data_set(data_path, need_unique)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        sample = self.data_set[index]

        data = sample['data']
        seq_info = deal_with_sample(data.split())
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
        seq_info = torch.LongTensor(seq_info)

        return (class_seq, api_seq, candidate_api_seq, seq_info), label

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


class APIHelperDataSet(Dataset):
    def __init__(self, data_path, api_dict, class_dict, class_to_api_dict, max_len, need_unique=False):
        self.api_dict = api_dict
        self.class_dict = class_dict
        self.max_len = max_len
        self.data_set = load_data_set(data_path, need_unique)
        self.api_len = len(self.api_dict)
        self.class_to_api_dict = class_to_api_dict

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        """pad the seq for each sample"""
        sample = self.data_set[index]

        data = sample['data']
        api_seq = []
        class_seq = []
        items = data.split()

        hole_loc = items.index('HOLE')
        hole_class = items[hole_loc - 1]

        seq_info = deal_with_sample(items)
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

        candidate_api_seq = self.class_to_api_dict[self.class_dict[hole_class] if hole_class in self.class_dict else 1]
        candidates = torch.zeros(self.api_len)
        candidates[candidate_api_seq] = 1
        if label == 1:
            candidates[label] = 1

        """list to tensor"""
        api_seq = torch.LongTensor(api_seq)
        class_seq = torch.LongTensor(class_seq)
        candidates = torch.FloatTensor(candidates)
        seq_info = torch.LongTensor(seq_info)

        return (class_seq, api_seq, candidates, seq_info), label


class NGramDataSet(Dataset):
    def __init__(self, data_path, api_dict, class_dict, k, need_unique=False):
        self.api_dict = api_dict
        self.class_dict = class_dict
        self.k = k
        self.data_set = load_data_set(data_path, need_unique)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        """pad the seq for each sample"""
        sample = self.data_set[index]

        data = sample['data']

        items = data.split()
        seq_info = deal_with_sample(items)

        hole_loc = items.index('HOLE')

        begin_index = hole_loc-1-2*self.k
        end_index = hole_loc+2*self.k + 1
        api_seq = []
        for i in range(begin_index, end_index, 2):
            if i < 0 or i > len(items)-2:
                api_seq.append('PAD')
            else:
                class_name = items[i]
                api_name = items[i+1]
                if api_name == 'HOLE':
                    continue
                api_seq.append(class_name + '.' + api_name)
        api_seq = [self.api_dict[x] if x in self.api_dict else 1 for x in api_seq]

        label = sample['label']
        label = self.api_dict[label] if label in self.api_dict else 1
        """list to tensor"""
        api_seq = torch.LongTensor(api_seq)
        seq_info = torch.LongTensor(seq_info)

        return (api_seq, seq_info), label


class StandardDataSet(Dataset):
    def __init__(self, data_path, api_dict, class_dict, max_len, need_unique=False):
        self.api_dict = api_dict
        self.class_dict = class_dict
        self.max_len = max_len
        self.data_set = load_data_set(data_path, need_unique)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        """pad the seq for each sample"""
        sample = self.data_set[index]

        data = sample['data']
        api_seq = []
        class_seq = []
        items = data.split()
        seq_info = deal_with_sample(items)
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
        seq_info = torch.LongTensor(seq_info)

        return (class_seq, api_seq, seq_info), label


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


def load_data_set(data_set_path, need_unique=False):
    data = []
    already_contains = []
    with open(data_set_path, 'r') as f:
        for line in f:
            line = line.strip()
            sample = dict()
            items = line.split(";")
            sample['data'] = items[0]
            sample['label'] = items[1]
            if need_unique:
                if line in already_contains:
                    continue
                already_contains.append(line)
            data.append(sample)
    return data


def deal_with_sample(api_seq):
    seq_len = int(len(api_seq) / 2)
    if seq_len > 10:
        seq_len = 10
    hole_loc = (api_seq.index('HOLE') + 1) / 2
    hole_loc = hole_loc / seq_len
    if hole_loc <= 1/5:
        hole_rate = 1
    elif hole_loc <= 2/5:
        hole_rate = 2
    elif hole_loc <= 3/5:
        hole_rate = 3
    elif hole_loc <= 4/5:
        hole_rate = 4
    else:
        hole_rate = 5
    class_set = set()
    for i, api in enumerate(api_seq):
        if i % 2 == 0:
            class_set.add(api)
    class_num = len(class_set)
    if class_num > 10:
        class_num = 10

    seq_info = [seq_len, hole_rate, class_num]
    return seq_info


def get_data_loaders(api_dict, class_dict, class_to_api_dict, args):
    train_data_set_path = args.data_dir + 'test_train/data_set_all.txt'
    valid_data_set_path = args.data_dir + 'test_valid/data_set_all.txt'
    if args.model == 'lstm':
        train_data_set = StandardDataSet(train_data_set_path,
                                         api_dict,
                                         class_dict,
                                         args.api_max_len)
        valid_data_set = StandardDataSet(valid_data_set_path,
                                         api_dict,
                                         class_dict,
                                         args.api_max_len)
    elif args.model == 'my_model' or args.model == 'my_model+':
        train_data_set = MyDataSet(train_data_set_path,
                                   api_dict,
                                   class_dict,
                                   class_to_api_dict,
                                   args.class_max_len,
                                   args.api_max_len)
        valid_data_set = MyDataSet(valid_data_set_path,
                                   api_dict,
                                   class_dict,
                                   class_to_api_dict,
                                   args.class_max_len,
                                   args.api_max_len)
    elif args.model == 'APIHelper' or args.model == 'APIHelper+':
        train_data_set = APIHelperDataSet(train_data_set_path,
                                          api_dict,
                                          class_dict,
                                          class_to_api_dict,
                                          args.api_max_len)
        valid_data_set = APIHelperDataSet(valid_data_set_path,
                                          api_dict,
                                          class_dict,
                                          class_to_api_dict,
                                          args.api_max_len)
    elif args.model == 'ngram':
        train_data_set = NGramDataSet(train_data_set_path,
                                      api_dict,
                                      class_dict,
                                      2)
        valid_data_set = NGramDataSet(valid_data_set_path,
                                      api_dict,
                                      class_dict,
                                      2)

    train_loader = DataLoader(dataset=train_data_set,
                              batch_size=args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_data_set,
                              batch_size=args.batch_size,
                              shuffle=False)
    return train_loader, valid_loader
