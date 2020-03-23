"""
generate training and test data set
"""
import copy
import random
from tqdm import tqdm


def generate_data_set(api_seq_path, save_path):
    data_set = dict()
    api_seq_data = load_api_seq(api_seq_path)
    labels = dict()
    for api_seq in tqdm(api_seq_data, desc='deal with api seq file'):
        samples = generate_sample(api_seq)
        for sample in samples:
            label = sample['label']
            if label not in labels:
                labels[label] = 0
            labels[label] += 1
            if labels[label] > 200:
                """
                if the label num > 200, 50% drop this sample and 50% random replace it
                """
                if random.randint(0, 1) % 2 == 0:
                    continue
                else:
                    label_samples = data_set[label]
                    replace_id = random.randint(0, len(label_samples)-1)
                    label_samples[replace_id] = sample
            else:
                if label not in data_set:
                    data_set[label] = []
                if sample not in data_set[label]:
                    data_set[label].append(sample)

    data_set = [sample for label in data_set for sample in data_set[label]]

    save_data_set(data_set, save_path)
    return


def generate_sample(api_seq):
    samples = []
    api_calls = api_seq.split()
    seq_len = len(api_calls)
    for i, api in enumerate(api_calls):
        if i % 2 != 0:
            sample = dict()
            sample['data'] = copy.deepcopy(api_calls)
            sample['data'][i] = 'HOLE'
            sample['label'] = api_calls[i-1]+'.'+api_calls[i]

            sample_without_suffix = copy.deepcopy(sample)
            sample_without_suffix['data'] = sample_without_suffix['data'][:i+1]

            if i != seq_len-1:
                samples.append(sample)
            if i != 1:
                samples.append(sample_without_suffix)
    return samples


def load_api_seq(api_seq_path):
    """
    load api seq from file
    """
    api_seq_data = []
    with open(api_seq_path, 'r') as f:
        for line in f:
            api_seq_data.append(line.strip())
    return api_seq_data


def save_data_set(data_set, save_path):
    with open(save_path + 'itext.txt', 'w') as f:
        for sample in data_set:
            f.write(' '.join(sample['data']) + ';' + sample['label'] + '\n')


if __name__ == '__main__':
    generate_data_set('../data/test/itext7-develop.txt', '../data/test/')
