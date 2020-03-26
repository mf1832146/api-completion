import torch
from torch.utils.data import DataLoader
import os
import module
from dataset import StandardDataSet, MyDataSet

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")
test_projects = ['Froyo_Email.txt', 'galaxy.txt', 'GridSphere.txt', 'itext.txt',
                     'jgit.txt', 'log4j.txt']


def load_model(args, api_vocab_size, class_vocab_size):
    model = module.make_model(args, api_vocab_size, class_vocab_size)
    checkpoint = torch.load(
            'train/models/my_model_1.0' +'prefix_my_model_1.0_647520.pth')
    model.load_state_dict(checkpoint)
    return model


def create_summary_writer(log_dir):
    writer = SummaryWriter(logdir=log_dir)
    return writer


def test(args, api_dict, class_dict, class_to_api_dict):
    writer = create_summary_writer('test_logs/' + args.model + '/')
    model = load_model(args, len(api_dict), len(class_dict))
    tt = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            ttt = 1
            for s in param.data.size():
                ttt *= s
            tt += ttt
    writer.add_scalar('total_parameter_num', tt, 0)

    for test_project in test_projects:
        project_path = args.data_dir + 'test/' + test_project
        if args.model == 'lstm':
            test_data_set = StandardDataSet(project_path,
                                            api_dict,
                                            class_dict,
                                            args.api_max_len)
            test_data_loader = DataLoader(dataset=test_data_set,
                                          batch_size=args.batch_size,
                                          shuffle=False)
        elif args.model == 'my_model':
            test_data_set = MyDataSet(project_path,
                                      api_dict,
                                      class_dict,
                                      class_to_api_dict,
                                      args.class_max_len,
                                      args.api_max_len)
            test_data_loader = DataLoader(dataset=test_data_set,
                                          batch_size=args.batch_size,
                                          shuffle=False)

        """top-k Acc"""
        model.eval()
        top_k_num = [0] * 10
        total_num = 0
        mrr = 0.0
        mrr_with_hole_loc = [0] * 5
        total_num_wih_hole_loc = [0] * 5
        mrr_with_seq_len = [0] * 10
        total_num_wih_seq_len = [0] * 10
        mrr_with_class_num = [0] * 10
        total_num_with_class_num = [0] * 10

        for i, data_batch in enumerate(test_data_loader):
            inputs, label = data_batch
            predicts = model(inputs)
            _, idx = torch.sort(predicts, dim=1, descending=True)
            #print(inputs[-1][:, 0][0])
            #print(idx[0])
            #print(inputs[-2][0])
            label = label.unsqueeze(1)
            hit_loc = (idx == label).nonzero()
            hit_loc = hit_loc[:, 1]
            for j in range(len(top_k_num)):
                top_k_num[j] += torch.sum(hit_loc <= j).cpu().data.numpy()

            seq_info = inputs[-1]
            seq_len = seq_info[:, 0]
            hole_rate = seq_info[:, 1]
            class_num = seq_info[:, 2]

            mrr_batch = 1 / (hit_loc.float() + 1)
            for j in range(10):
                mrr_with_seq_len[j] += torch.sum(mrr_batch * (seq_len == (j+1)).float()).data.numpy()
                mrr_with_class_num[j] += torch.sum(mrr_batch * (class_num == (j+1)).float()).data.numpy()
                total_num_wih_seq_len[j] += torch.sum(seq_len == (j+1)).data.numpy()
                total_num_with_class_num[j] += torch.sum(class_num == (j+1)).data.numpy()
                if j < 5:
                    mrr_with_hole_loc[j] += torch.sum(mrr_batch * (hole_rate == (j+1)).float()).data.numpy()
                    total_num_wih_hole_loc[j] += torch.sum(hole_rate == (j+1)).data.numpy()
            mrr += torch.sum(mrr_batch).data.numpy()
            total_num += label.size(0)

        for i in range(len(top_k_num)):
            top_k_i = top_k_num[i] / total_num
            writer.add_scalar(test_project + '/' + ' top-k', top_k_i, i+1)
        """MRR"""
        mrr = mrr / total_num
        writer.add_scalar(test_project + '/' + 'mrr', mrr, 0)
        """seq_len_mrr"""
        for i in range(10):
            if total_num_wih_seq_len[i] == 0:
                total_num_wih_seq_len[i] = 1
            mrr_with_seq_len_i = mrr_with_seq_len[i] / total_num_wih_seq_len[i]
            writer.add_scalar(test_project + '/MRR with seq len', mrr_with_seq_len_i, i+1)
        """hole_loc_mrr"""
        for i in range(5):
            if total_num_wih_hole_loc[i] == 0:
                total_num_wih_hole_loc[i] = 1
            mrr_with_hole_loc_i = mrr_with_hole_loc[i] / total_num_wih_hole_loc[i]
            writer.add_scalar(test_project + '/MRR with hole loc', mrr_with_hole_loc_i, i+1)
        """class_num_mrr"""
        for i in range(10):
            if total_num_with_class_num[i] == 0:
                total_num_with_class_num[i] = 1
            mrr_with_class_num_i = mrr_with_class_num[i] / total_num_with_class_num[i]
            writer.add_scalar(test_project + '/MRR with class num', mrr_with_class_num_i, i+1)
    writer.close()
