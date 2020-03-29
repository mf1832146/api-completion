"""
n-gram and hapi
"""
import numpy as np
from hmmlearn import hmm
import pickle
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


def test(api_seq, hole_class, save_path):
    with open(save_path + hole_class + '.pkl', 'rb') as f:
        out = pickle.load(f)
        api_list = out['api_list']
        model = out['model']
    observation_seq = []
    items = api_seq.split()
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
    for i in range(len(api_list)):
        observation_seq[hole_loc] = i
        score = model.score(observation_seq)
        scores.append(score)
    return score


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
    train(api_seq_path, output, save_path)
