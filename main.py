import argparse
import module
import train
import logging


def parse():
    parser = argparse.ArgumentParser(description='api completion')
    parser.add_argument('-model_dir', default='./train_model', help='output model weight dir')
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-max_epoch', type=int, default=250)
    parser.add_argument('-data_dir', default='./data/')

    parser.add_argument('-api_dict', default='api_dict.txt')
    parser.add_argument('-class_dict', default='class_dict.txt')
    parser.add_argument('-load',  action='store_true', help='load pretrained model')
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-log_dir', type=str, default='tb_logs/')

    parser.add_argument('-class_max_len', type=int, default=20, help='the max number of classes in an API seq')
    parser.add_argument('-api_max_len', type=int, default=20, help='the max number of APIs in a class')

    parser.add_argument('-api_emb_dim', default=128)
    parser.add_argument('-class_emb_dim', default=128)
    parser.add_argument('-hidden_size', default=256)
    parser.add_argument('-dropout', default=0.3)
    parser.add_argument('-lr', default=0.001)

    args = parser.parse_args()
    return args


def load_api_dict(api_dict_path):
    api_dict = {"PAD": 0, "UNK": 1, "HOLE": 2}
    ids = 3
    with open(api_dict_path, 'r') as f:
        for line in f:
            api_dict[line.strip()] = ids
            ids += 1
    return api_dict


def load_class_dict(class_dict_path):
    class_dict = {"PAD": 0, "UNK": 1}
    ids = 2
    with open(class_dict_path, 'r') as f:
        for line in f:
            class_dict[line.strip()] = ids
            ids += 1
    return class_dict


if __name__ == '__main__':
    args = parse()

    # Setup engine logger
    logger = logging.getLogger("ignite.engine.engine.Engine")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    api_dict = load_api_dict(args.data_dir + args.api_dict)
    class_dict = load_class_dict(args.data_dir + args.class_dict)
    model = module.make_model(args, len(api_dict), len(class_dict))
    solver = train.Solver(args, model, api_dict, class_dict)
    solver.train()
