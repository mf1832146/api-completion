"""
n-gram and hapi
"""
import numpy as np
from hmmlearn import hmm
import pickle

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os


def create_class_hmm(api_seq, save_path, class_name, api_list):
    n_states = 8
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.001)
    api_seq = [[[int(x)] for x in item.split()] for item in api_seq]
    if len(api_seq) > 1:
        X = np.concatenate(api_seq)
        lengths = [len(x) for x in api_seq]
        model.fit(X, lengths)

        # save model
        save_model = {'model': model, 'api_list': api_list}
        with open(save_path + class_name + '.pkl', 'wb') as f:
            pickle.dump(save_model, f)


def read_api_seq(api_seq_path, class_to_api_dict):
    data = dict()
    with open(api_seq_path, 'r') as f:
        for line in f:
            class_api_seq = dict()
            items = line.split()
            for i in range(0, len(items)-1, 2):
                class_name = items[i]
                api = class_name + '.' + items[i+1]
                if class_name not in class_api_seq:
                    class_api_seq[class_name] = []
                class_api_seq[class_name].append(api)
            for class_name in class_api_seq:
                if len(class_api_seq[class_name]) < 2:
                    continue
                if class_name not in data:
                    data[class_name] = set()
                data[class_name].add(' '.join([str(class_to_api_dict[class_name].index(x)) for x in class_api_seq[class_name]]))
    return data


def train(api_seq_path, class_to_api_dict, save_path):
    files = os.listdir(save_path)
    data = read_api_seq(api_seq_path, class_to_api_dict)
    for class_name in class_to_api_dict:
        if class_name == 'UNK':
            continue
        if class_name not in data:
            continue
        if class_name + '.pkl' in files:
            print('already processed')
            continue
        print('deal with ', class_name)
        print('seq len:', len(data[class_name]))
        create_class_hmm(data[class_name], save_path, class_name, class_to_api_dict[class_name])


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


def test_hmm(test_path, model_save_path):
    for file in os.listdir(test_path):
        if file == '.DS_Store':
            continue
        print(file)
        top_k_num = [0] * 10
        total_num = 0
        total_mrr = 0.0
        mrr_with_hole_loc = [0.] * 5
        total_num_wih_hole_loc = [0] * 5
        mrr_with_seq_len = [0.] * 10
        total_num_wih_seq_len = [0] * 10
        mrr_with_class_num = [0.] * 10
        total_num_with_class_num = [0] * 10
        with open(test_path + file, 'r') as f:
            for line in f:
                line = line.strip()
                items = line.split(";")
                data = items[0]
                label = items[1]
                hole_class = label.split('.')[0]
                hit_loc, seq_info = test(data, hole_class, model_save_path, label)
                seq_len, hole_rate, class_num = seq_info

                total_num += 1
                total_num_with_class_num[class_num-1] += 1
                total_num_wih_hole_loc[hole_rate-1] += 1
                total_num_wih_seq_len[int(seq_len-1)] += 1

                if hit_loc == -1: # not hit
                    continue
                else:
                    for i in range(hit_loc, 10):
                        top_k_num[i] += 1
                    mrr = 1 / (hit_loc + 1)
                    mrr_with_seq_len[seq_len-1] += mrr
                    mrr_with_class_num[class_num-1] += mrr
                    mrr_with_hole_loc[hole_rate-1] += mrr
                    total_mrr += mrr

        # write to tensorboard
        writer = SummaryWriter(logdir='../test_logs/hmm/')
        for i in range(len(top_k_num)):
            top_k_i = top_k_num[i] / total_num
            writer.add_scalar(file + '/' + ' top-k', top_k_i, i + 1)
        """MRR"""
        total_mrr = total_mrr / total_num
        writer.add_scalar(file + '/' + 'mrr', total_mrr, 0)
        """seq_len_mrr"""
        for i in range(10):
            if total_num_wih_seq_len[i] == 0:
                total_num_wih_seq_len[i] = 1
            mrr_with_seq_len_i = mrr_with_seq_len[i] / total_num_wih_seq_len[i]
            writer.add_scalar(file + '/MRR with seq len', mrr_with_seq_len_i, i + 1)
        """hole_loc_mrr"""
        for i in range(5):
            if total_num_wih_hole_loc[i] == 0:
                total_num_wih_hole_loc[i] = 1
            mrr_with_hole_loc_i = mrr_with_hole_loc[i] / total_num_wih_hole_loc[i]
            writer.add_scalar(file + '/MRR with hole loc', mrr_with_hole_loc_i, i + 1)
        """class_num_mrr"""
        for i in range(10):
            if total_num_with_class_num[i] == 0:
                total_num_with_class_num[i] = 1
            mrr_with_class_num_i = mrr_with_class_num[i] / total_num_with_class_num[i]
            writer.add_scalar(file + '/MRR with class num', mrr_with_class_num_i, i + 1)
    writer.close()


def test(api_seq, hole_class, save_path, label):
    observation_seq = []
    items = api_seq.split()
    seq_info = deal_with_sample(items)
    class_model_list = os.listdir(save_path)
    if hole_class + '.pkl' not in class_model_list:
        return -1, seq_info
    with open(save_path + hole_class + '.pkl', 'rb') as f:
        out = pickle.load(f)
        api_list = out['api_list']
        model = out['model']

    for i in range(0, len(items)-1, 2):
        class_name = items[i]
        api_name = items[i+1]
        if api_name == 'HOLE':
            observation_seq.append(-1)
        else:
            api_name = class_name + '.' + api_name
            if api_name in api_list:
                observation_seq.append(api_list.index(api_name))

    scores = []
    hole_loc = observation_seq.index(-1)
    label_id = api_list.index(label) if label in api_list else -1
    for i in range(len(api_list)):
        observation_seq[hole_loc] = i
        try:
            score = model.score([[x] for x in observation_seq])
        except Exception:
            return -1, seq_info
        scores.append(score)

    # sort by score
    idx = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    idx = [x[0] for x in idx][:10]
    hit_id = idx.index(label_id) if label_id in idx else -1

    return hit_id, seq_info


def load_dict(api_dict_path):
    api_dict = {"PAD": 0, "UNK": 1, "HOLE": 2}
    class_dict = {"PAD": 0, "UNK": 1}
    class_to_api_dict = {1: [1]}
    api_ids = 3
    class_ids = 2
    with open(api_dict_path, 'r') as f:
        for line in f:
            line = line.strip()

            api_dict[line] = api_ids

            class_name = line.split('.')[0]
            if class_name not in class_dict:
                class_dict[class_name] = class_ids
                class_ids += 1

            class_id = class_dict[class_name]
            if class_id not in class_to_api_dict:
                class_to_api_dict[class_id] = []
            class_to_api_dict[class_id].append(api_ids)

            api_ids += 1
    return api_dict, class_dict, class_to_api_dict


if __name__ == '__main__':
    api_seq_path = '../data/api_seq_all.txt'
    save_path = '../models/hmm/'
    api_dict, class_dict, class_to_api_dict = load_dict('../data/api_dict.txt')
    class_dict = {v:k for k,v in class_dict.items()}
    api_dict = {v:k for k,v in api_dict.items()}
    output = dict()
    for class_id in class_to_api_dict:
        output[class_dict[class_id]] = []
        for api_id in class_to_api_dict[class_id]:
            output[class_dict[class_id]].append(api_dict[api_id])
    # train(api_seq_path, output, save_path)
    model_save_path = '../models/hmm/'
    test_hmm('../data/test/', model_save_path)
