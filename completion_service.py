from flask import Flask

import module
from dataset import pad_seq
from main import parse, load_dict
from test import load_model
from flask import request
import torch
import torch.nn.functional as F

from utils import deal_with_code, deal_with_code_for_APIHelper

app = Flask(__name__)


def load_model(args, api_vocab_size, class_vocab_size):
    model = module.make_model(args, api_vocab_size, class_vocab_size)
    checkpoint = torch.load(
            'models/' + args.model + '/' +  args.model + '_' + args.model_path + '.pth', map_location='cpu')
    model.load_state_dict(checkpoint)
    return model


class ModelParams:
    def __init__(self):
        self.data_dir = './data/'
        self.api_dict = 'api_dict.txt'
        self.model_path = '20'
        self.model = 'my_model'
        self.api_emb_dim = 128
        self.class_emb_dim = 128
        self.hidden_size = 256
        self.dropout = 0.3


def init():
    args = ModelParams()

    api_dict, class_dict, class_to_api_dict = load_dict(args.data_dir + args.api_dict)
    model = load_model(args, len(api_dict), len(class_dict))
    model.eval()
    return model, api_dict, class_dict, class_to_api_dict


my_model, my_api_dict, my_class_dict, my_class_to_api_dict = init()
my_id_to_api_dict = {item: key for key, item in my_api_dict.items()}


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/complete/', methods=['POST'])
def complete():
    code = request.form['username']
    inputs = deal_with_code(code, my_class_dict, my_api_dict, my_class_to_api_dict)
    _, _, candidate_api_seq, _ = inputs
    predicts = my_model(inputs)
    predicts = F.softmax(predicts, dim=1)

    scores, idx = torch.sort(predicts, dim=1, descending=True)

    candidate_api_seq = candidate_api_seq.squeeze(0).tolist()
    scores = scores.squeeze(0).tolist()
    idx = idx.squeeze(0).tolist()
    results = []
    for i in range(len(candidate_api_seq)):
        result = dict()
        result['score'] = scores[i]
        if candidate_api_seq[idx[i]] == 0:
            break
        result['API'] = my_id_to_api_dict[candidate_api_seq[idx[i]]]
        results.append(result)
    return {'results': results}


@app.route('/complete_with_apihelper/', methods=['POST'])
def complete_with_apiHelper():
    code = request.form['username']
    inputs = deal_with_code_for_APIHelper(code, my_class_dict, my_api_dict, my_class_to_api_dict, 10, len(my_api_dict))
    predicts = my_model(inputs)
    predicts = F.softmax(predicts, dim=1)
    scores, idx = torch.sort(predicts, dim=1, descending=True)
    predicts = predicts.squeeze(0).tolist()
    scores = scores.squeeze(0).tolist()
    idx = idx.squeeze(0).tolist()
    results = []
    for i in range(len(scores)):
        if scores[i] < 1e-3:
            break
        result = dict()
        result['score'] = scores[i]
        result['API'] = my_id_to_api_dict[idx[i]]
        results.append(result)
    return {'results': results}
