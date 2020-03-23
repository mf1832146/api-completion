import argparse
from solver import Solver


def parse():
    parser = argparse.ArgumentParser(description='api completion')
    parser.add_argument('-model_dir', default='./train_model', help='output model weight dir')
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-num_step', type=int, default=250)
    parser.add_argument('-data_dir', default='../data/')
    parser.add_argument('-train_data_dir', default='../data/train/data_set.txt')
    parser.add_argument('-api_dict', default='api_dict.txt')
    parser.add_argument('-class_dict', default='class_dict.txt')
    parser.add_argument('-load',  action='store_true', help='load pretrained model')
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-test', action='store_true')

    parser.add_argument('-class_max_len', type=int, default=10, help='the max number of classes in an API seq')
    parser.add_argument('-api_max_len', type=int, default=20, help='the max number of APIs in a class')

    parser.add_argument('-api_emb_dim', default=64)
    parser.add_argument('-class_emb_dim', default=64)
    parser.add_argument('-hidden_size', default=256)
    parser.add_argument('-dropout', default=0.5)
    parser.add_argument('-lr', default=0.001)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)

    if args.train:
        solver.train()
    elif args.test:
        solver.test()
