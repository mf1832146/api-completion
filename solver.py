from torch.utils.data import DataLoader

import utils
import os
import torch
from torch import optim
import torch.nn as nn
from dataset import StandardAPIDataSet
from module import StandardLSTM
from train import SimpleLossCompute
import time


class Solver:
    def __init__(self, args):
        self.args = args
        self.use_gpu = False
        self.model_dir = utils.make_save_dir(args.model_dir)

        self.api_dict = utils.load_api_dict(self.args.data_dir + self.args.api_dict)
        self.class_dict = utils.load_class_dict(self.args.data_dir + self.args.class_dict)
        self.model = StandardLSTM(api_vocab=len(self.api_dict), api_emb_dim=self.args.api_emb_dim,
                                  class_vocab=len(self.class_dict), class_emb_dim=self.args.class_emb_dim,
                                  hidden_size=self.args.hidden_size,
                                  dropout=self.args.dropout)
        if self.use_gpu:
            self.model = self.model.cuda()

    def train(self):
        if self.args.load:
            path = os.path.join(self.model_dir, 'model.pth')
            self.model.load_state_dict(torch.load(path)['state_dict'])

        tt = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                ttt = 1
                for s in param.data.size():
                    ttt *= s
                tt += ttt
        print('total param num:', tt)

        print('Loading training data...')
        train_data_set = StandardAPIDataSet(self.args.train_data_dir,
                                            self.api_dict,
                                            self.class_dict,
                                            self.args.api_max_len,
                                            self.use_gpu)
        train_loader = DataLoader(dataset=train_data_set, batch_size=self.args.batch_size, shuffle=True)
        print('load training data finished')

        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(self.model.parameters(), lr=self.args.lr)
        loss_compute = SimpleLossCompute(criterion, opt)

        total_loss = []
        for step in range(self.args.num_step):
            self.model.train()

            start = time.time()
            step_loss = run_epoch(step, train_loader, self.model, loss_compute)
            elapsed = time.time() - start
            print('----------epoch: %d end, total loss= %f , train_time= %f Sec -------------' % (
            step, step_loss, elapsed))
            total_loss.append(step_loss)
            print('saving!!!!')

            model_name = 'model.pth'
            state = {'epoch': step, 'state_dict': self.model.state_dict()}
            torch.save(state, os.path.join(self.model_dir, model_name))

        print('training process end, total_loss is =', total_loss)

    def test(self):
        return


def run_epoch(epoch, data_iter, model, loss_compute):
    total_loss = 0
    total_num = 0
    for i, data_batch in enumerate(data_iter):
        class_seq, api_seq, label = data_batch
        batch_size = class_seq.size(0)
        out = model.forward(class_seq, api_seq)
        loss = loss_compute(out, label)
        total_loss += loss * batch_size
        total_num += batch_size
        if i % 50 == 1:
            print("Epoch %d Step: %d Loss: %f" %
                  (epoch, i, loss))
    return total_loss / total_num






