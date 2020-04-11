import collections
import torch

from dataset import pad_seq



def deal_with_code_for_APIHelper(code, class_dict, api_dict,class_to_api_dict,  max_len, api_len):
    api_seq = []
    class_seq = []
    items = code.split()

    hole_loc = items.index('HOLE')
    hole_class = items[hole_loc - 1]

    for i, api in enumerate(items):
        if i % 2 == 0:
            class_seq.append(api)
        else:
            if api == "HOLE":
                api_seq.append('HOLE')
            else:
                api_seq.append(items[i - 1] + '.' + api)

    class_seq, api_seq = pad_seq(class_seq, api_seq, max_len)
    api_seq = [api_dict[x] if x in api_dict else 1 for x in api_seq]
    class_seq = [class_dict[x] if x in class_dict else 1 for x in class_seq]

    candidate_api_seq = class_to_api_dict[class_dict[hole_class] if hole_class in class_dict else 1]
    candidates = torch.zeros(api_len)
    candidates[candidate_api_seq] = 1

    """list to tensor"""
    api_seq = torch.LongTensor(api_seq)
    class_seq = torch.LongTensor(class_seq)
    candidates = torch.FloatTensor(candidates)

    class_seq = class_seq.unsqueeze(0)
    api_seq = api_seq.unsqueeze(0)
    candidates = candidates.unsqueeze(0)

    return class_seq, api_seq, candidates, None


def deal_with_code(code, class_dict, api_dict, class_to_api_dict, max_class_len=10, max_api_len=10, max_candidate_api_len=200):
    class_seq, api_seq, hole_class = get_sample(code, max_class_len)
    for i in range(len(api_seq)):
        if len(api_seq[i]) < max_api_len:
            api_seq[i].extend(['PAD'] * max_api_len)
            api_seq[i] = api_seq[i][:max_api_len]
    if len(class_seq) < max_class_len:
        class_seq.extend(['PAD'] * max_class_len)
        class_seq = class_seq[:max_class_len]
        api_seq.extend([['PAD'] * max_api_len] * max_class_len)
        api_seq = api_seq[:max_class_len]

    class_seq = [class_dict[x] if x in class_dict else 1 for x in class_seq]
    api_seq = [[api_dict[x] if x in api_dict else 1 for x in y] for y in api_seq]

    candidate_api_seq = class_to_api_dict[class_dict[hole_class] if hole_class in class_dict else 1]
    if len(candidate_api_seq) < max_candidate_api_len:
        candidate_api_seq.extend([0] * max_candidate_api_len)
        candidate_api_seq = candidate_api_seq[:max_candidate_api_len]
    else:
        candidate_api_seq = candidate_api_seq[:max_candidate_api_len]

    class_seq = torch.LongTensor(class_seq)
    api_seq = torch.LongTensor(api_seq)
    candidate_api_seq = torch.LongTensor(candidate_api_seq)

    class_seq = class_seq.unsqueeze(0)
    api_seq = api_seq.unsqueeze(0)
    candidate_api_seq = candidate_api_seq.unsqueeze(0)

    return class_seq, api_seq, candidate_api_seq, None


def get_sample(api_seq, max_class_len):
    items = api_seq.split()
    hole_loc = items.index('HOLE')
    pre_class_seq, pre_api_seq = deal_with_sub_str(items[:hole_loc - 1], True)
    hole_class = items[hole_loc - 1]
    hole_api = 'HOLE'
    suffix_class_seq, suffix_api_seq = deal_with_sub_str(items[hole_loc + 1:], False)

    if len(pre_class_seq) + len(suffix_class_seq) > max_class_len - 1:
        if len(pre_class_seq) < max_class_len / 2:
            suffix_seq_len = max_class_len - len(pre_class_seq) - 1
            suffix_class_seq = suffix_class_seq[:suffix_seq_len]
            suffix_api_seq = suffix_api_seq[:suffix_seq_len]
        elif len(suffix_class_seq) < max_class_len / 2:
            pre_seq_len = max_class_len - len(suffix_class_seq) - 1
            pre_class_seq = pre_class_seq[-1 * pre_seq_len:]
            pre_api_seq = pre_api_seq[-1 * pre_seq_len:]
        else:
            pre_len = int(max_class_len / 2)
            suffix_len = max_class_len - 1 - pre_len
            suffix_class_seq = suffix_class_seq[:suffix_len]
            suffix_api_seq = suffix_api_seq[:suffix_len]
            pre_class_seq = pre_class_seq[-1 * pre_len:]
            pre_api_seq = pre_api_seq[-1 * pre_len:]

    pre_class_seq.append(hole_class)
    pre_class_seq.extend(suffix_class_seq)
    pre_api_seq.append([hole_api])
    pre_api_seq.extend(suffix_api_seq)
    return pre_class_seq, pre_api_seq, hole_class


def deal_with_sub_str(sub_seq, is_pre, max_api_len=10):
    class_seq = []
    api_seq = []
    ordered_dict = collections.OrderedDict()
    for i in range(len(sub_seq) - 1, 0, -2):
        class_name = sub_seq[i - 1]
        api_name = class_name + '.' + sub_seq[i]
        if class_name not in ordered_dict:
            ordered_dict[class_name] = []
        ordered_dict[class_name].insert(0, api_name)

        if len(ordered_dict[class_name]) > max_api_len:
            seq = ordered_dict[class_name]
            if is_pre:
                ordered_dict[class_name] = seq[-max_api_len:]
            else:
                ordered_dict[class_name] = seq[:max_api_len]
    for key, item in ordered_dict.items():
        class_seq.insert(0, key)
        api_seq.insert(0, item)
    return class_seq, api_seq
