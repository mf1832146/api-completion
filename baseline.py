"""
n-gram and hapi
"""
import numpy as np
from hmmlearn import hmm
import pickle
from tqdm import tqdm


def create_class_hmm(api_seq, save_path, class_name, api_list):
    n_states = 8
    observation = []
    for item in api_seq:
        seq = []
        for api in item:
            if api in api_list:
                seq.append([api_list.index(api)])
        if len(seq) <= 2:
            continue
        observation.append(seq)
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.001)
    X = np.concatenate(observation)
    lengths = [len(x) for x in observation]
    model.fit(X, lengths)

    # save model
    save_model = {'model': model, 'api_list': api_list}
    with open(save_path + class_name + '.pkl', 'wb') as f:
        pickle.dump(save_model, f)


def read_api_seq(api_seq_path):
    data = []
    with open(api_seq_path, 'r') as f:
        for line in f:
            api_seq = []
            items = line.split()
            for i in range(0, len(items)-1, 2):
                api = items[i] + '.' + items[i+1]
                api_seq.append(api)
            data.append(api_seq)
    return data


def train(api_seq_path, class_to_api_dict, save_path):
    api_seq = read_api_seq(api_seq_path)
    for class_name in tqdm(class_to_api_dict):
        if class_name == 'UNK':
            continue
        create_class_hmm(api_seq, save_path, class_name, class_to_api_dict[class_name])


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
    api_seq_path = './data/api_seq_all.txt'
    save_path = './models/hmm/'
    api_dict, class_dict, class_to_api_dict = load_dict('./data/api_dict.txt')
    class_dict = {v:k for k,v in class_dict.items()}
    api_dict = {v:k for k,v in api_dict.items()}
    output = dict()
    for class_id in class_to_api_dict:
        output[class_dict[class_id]] = []
        for api_id in class_to_api_dict[class_id]:
            output[class_dict[class_id]].append(api_dict[api_id])
    train(api_seq_path, output, save_path)
