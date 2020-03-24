from torch.utils.data import Dataset, DataLoader
import torch


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


def get_data_loaders(api_dict, class_dict, args):
    train_data_set = StandardDataSet(args.data_dir + 'train/data_set.txt',
                                     api_dict,
                                     class_dict,
                                     args.api_max_len)
    train_loader = DataLoader(dataset=train_data_set,
                              batch_size=args.batch_size,
                              shuffle=True)
    valid_data_set = StandardDataSet(args.data_dir + 'valid/data_set.txt',
                                     api_dict,
                                     class_dict,
                                     args.api_max_len)
    valid_loader = DataLoader(dataset=valid_data_set,
                              batch_size=args.batch_size,
                              shuffle=False)
    return train_loader, valid_loader
