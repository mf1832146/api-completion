from torch.utils.data import Dataset
import utils
import torch


class StandardAPIDataSet(Dataset):
    """
    the API call seq data set for standard lstm and APIHelper
    """
    def __init__(self, data_dir, api_dict, class_dict, max_len, use_gpu):
        print('Loading data...')
        self.train_data = utils.load_data_set(data_dir)
        self.api_dict = api_dict
        self.class_dict = class_dict
        print('Load data finished...')
        self.max_len = max_len
        self.use_gpu = use_gpu

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        """pad the seq for each sample"""
        sample = self.train_data[index]

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
                    api_seq.append(items[i-1] + '.' + api)

        class_seq, api_seq = utils.pad_seq(class_seq, api_seq, self.max_len)
        api_seq = [self.api_dict[x] if x in self.api_dict else 1 for x in api_seq]
        class_seq = [self.class_dict[x] if x in self.class_dict else 1 for x in class_seq]

        label = sample['label']
        label = [self.api_dict[label] if label in self.api_dict else 1]

        """list to tensor"""
        api_seq = torch.LongTensor(api_seq)
        class_seq = torch.LongTensor(class_seq)
        label = torch.LongTensor(label)

        if self.use_gpu:
            api_seq = api_seq.cuda()
            class_seq = class_seq.cuda()
            label = label.cuda()

        return class_seq, api_seq, label
