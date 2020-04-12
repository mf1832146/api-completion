import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable

HOLE = 2


class MyModel(nn.Module):
    def __init__(self, api_vocab, api_emb_dim,
                 class_vocab, class_emb_dim,
                 hidden_size,
                 dropout=0.0,
                 class_attn=False):
        super(MyModel, self).__init__()
        self.api_emb_layer = nn.Embedding(api_vocab, api_emb_dim, padding_idx=0)
        self.class_emb_layer = nn.Embedding(class_vocab, class_emb_dim, padding_idx=0)

        self.api_emb_dim = api_emb_dim
        self.class_emb_dim = class_emb_dim
        self.class_rnn = nn.LSTM(input_size=api_emb_dim+class_emb_dim,
                                 hidden_size=hidden_size,
                                 batch_first=True)
        self.func_rnn = nn.LSTM(input_size=hidden_size,
                                hidden_size=int(hidden_size/2),
                                bidirectional=True,
                                batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=api_emb_dim)
        if class_attn:
            self.class_attn = Class2ClassAttn(class_emb_dim, hidden_size, dropout)
        else:
            self.class_attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        """
        class_seq [batch_size, max_class_len]
        api_seq [batch_size, max_class_len, max_api_len]
        candidate_api_seq [batch_size, max_candidate_len]
        """
        class_seq, api_seq, candidate_api_seq, _ = inputs
        batch_size, max_class_len, max_api_len = api_seq.size()
        hole_loc = (api_seq == 2).nonzero()[:, 1]
        api_seq = api_seq.view(-1, max_api_len)
        api_len = torch.sum(api_seq != 0, dim=1)
        api_len = (api_len - 1).tolist()

        class_emb = self.class_emb_layer(class_seq)
        api_emb = self.api_emb_layer(api_seq)
        api_class_emb = class_emb.view(batch_size * max_class_len, -1)
        api_class_emb = api_class_emb.unsqueeze(1).repeat(1, max_api_len, 1)
        api_emb = torch.cat((api_class_emb, api_emb), dim=-1) * math.sqrt(self.class_emb_dim + self.api_emb_dim)
        class_emb = class_emb * math.sqrt(self.class_emb_dim)
        candidate_api_emb = self.api_emb_layer(candidate_api_seq) * math.sqrt(self.api_emb_dim)

        class_output, _ = self.class_rnn(api_emb)
        # batch_size*max_len, max_len, max hidden_size
        class_output = class_output[range(len(api_len)), api_len]
        class_output = class_output.view(batch_size, max_class_len, -1)
        if self.class_attn is not None:
            mask = class_seq == 0
            target_class_emb = class_emb[range(len(hole_loc)), hole_loc]
            scores = self.class_attn(class_emb, target_class_emb, mask)
            scores = scores.unsqueeze(2)
            class_output = torch.mul(scores, class_output)

        class_len = torch.sum(class_seq != 0, dim=1)
        _, idx_sort = torch.sort(class_len, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        class_output = class_output.index_select(0, idx_sort)
        class_len = list(class_len[idx_sort])
        class_output_packed = nn.utils.rnn.pack_padded_sequence(input=class_output, lengths=class_len, batch_first=True)

        func_output_packed, _ = self.func_rnn(class_output_packed)

        func_output = nn.utils.rnn.pad_packed_sequence(func_output_packed, batch_first=True)
        func_output = func_output[0].index_select(0, idx_unsort)
        func_output = self.dropout(func_output)

        output = func_output[range(len(hole_loc)), hole_loc]

        output = F.tanh(self.linear(output))
        output = torch.sum(torch.mul(output.unsqueeze(1), candidate_api_emb), dim=2)
        output = output.masked_fill(candidate_api_seq == 0, -1e9)
        return output


class NGram(nn.Module):
    def __init__(self, api_vocab, api_emb_dim, n, dropout):
        super(NGram, self).__init__()
        self.api_emb = nn.Embedding(api_vocab, api_emb_dim, padding_idx=0)
        self.linear = nn.Linear(api_emb_dim * n, api_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        api_seq, _ = inputs
        batch_size = api_seq.size(0)
        api_emb = self.api_emb(api_seq)
        api_emb = api_emb.view(batch_size, -1)
        api_emb = self.dropout(api_emb)
        outputs = self.linear(api_emb)

        return outputs


class APIHelper(nn.Module):
    def __init__(self, api_vocab, api_emb_dim,
                 class_vocab, class_emb_dim,
                 hidden_size,
                 dropout=0.0,
                 class_attn=False):
        super(APIHelper, self).__init__()
        self.emb_layer = APIEmbLayer(api_vocab, api_emb_dim, class_vocab, class_emb_dim)
        self.rnn = nn.LSTM(input_size=api_emb_dim + class_emb_dim,
                           hidden_size=hidden_size)
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=api_vocab)
        self.dropout = nn.Dropout(dropout)
        if class_attn:
            self.class_attn = ClassAttn(class_emb_dim, hidden_size, dropout)
        else:
            self.class_attn = None

    def forward(self, inputs):
        class_seq, api_seq, candidate_api_seq, _ = inputs

        hole_loc = (api_seq == 2).nonzero()[:, 1]

        class_emb, api_emb = self.emb_layer(class_seq, api_seq)
        output, _ = self.rnn(api_emb.permute([1, 0, 2]))
        output = output.permute([1, 0, 2])
        if self.class_attn is None:
            output = output[range(len(hole_loc)), hole_loc]
        else:
            class_emb = class_emb[range(len(hole_loc)), hole_loc]
            mask = api_seq == 0
            output = self.class_attn(class_emb, output, mask)
        output = self.dropout(output)
        output = self.linear(output)
        mask = Variable(candidate_api_seq == 0, requires_grad=False)
        output = output.masked_fill(mask, -1e9)
        return output


class Class2ClassAttn(nn.Module):
    def __init__(self, emb_dim, hidden_size, dropout):
        super(Class2ClassAttn, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.W_input = nn.Linear(self.emb_dim + self.emb_dim, 1)

    def forward(self, x, target_class_emb, mask=None):
        """
        :param class_emb: [batch_size, max_class_len, emb_dim]
        :param target_class_emb: [batch_size, emb_dim]
        :param mask:
        :return:
        """
        batch_size, max_class_len, _ = x.size()
        target_class_emb = target_class_emb.unsqueeze(1).repeat(1, max_class_len, 1)
        query = torch.cat((target_class_emb, x), dim=-1)
        query = query.view(-1, self.emb_dim + self.emb_dim)
        scores = F.sigmoid(self.W_input(query))
        scores = self.dropout(scores)
        scores = scores.view(batch_size, max_class_len)
        scores = scores.masked_fill(mask, 0)
        return scores


class ClassAttn(nn.Module):
    def __init__(self, emb_dim, hidden_size, dropout):
        super(ClassAttn, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.w_attn = nn.Linear(emb_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, class_emb, h, mask=None):
        """
        :param class_emb: [batch_size, emb_dim]
        :param h: [batch_size, max_len, hidden_size]
        :param mask: [batch_size, max_len]
        :return:
        """
        max_len = h.size(1)
        query = self.w_attn(class_emb)
        query = query.unsqueeze(1)

        # scores shape [batch_size, max_len]
        scores = torch.bmm(query, h.transpose(1, 2)) / math.sqrt(max_len)
        scores = scores.squeeze(1)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        p_attn = p_attn.unsqueeze(1)

        outputs = torch.bmm(p_attn, h).squeeze(1)

        return outputs


class StandardLSTM(nn.Module):
    def __init__(self, api_vocab, api_emb_dim,
                 class_vocab, class_emb_dim,
                 hidden_size,
                 dropout=0.0):
        super(StandardLSTM, self).__init__()
        self.emb_layer = nn.Embedding(api_vocab, api_emb_dim+class_emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(input_size=api_emb_dim+class_emb_dim,
                           hidden_size=hidden_size)
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=api_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        class_seq, api_seq, _ = inputs
        api_len = torch.sum(api_seq != 0, dim=1)
        api_len = (api_len - 1).tolist()
        api_emb = self.emb_layer(api_seq)
        output, _ = self.rnn(api_emb.permute([1, 0, 2]))
        output = output.permute([1, 0, 2])
        output = output[range(len(api_len)), api_len]
        output = self.dropout(output)
        output = self.linear(output)

        return output


class APIEmbLayer(nn.Module):
    def __init__(self, api_vocab, api_emb_dim,
                 class_vocab, class_emb_dim):
        super(APIEmbLayer, self).__init__()
        self.api_emb_layer = nn.Embedding(api_vocab, api_emb_dim, padding_idx=0)
        self.class_emb_layer = nn.Embedding(class_vocab, class_emb_dim, padding_idx=0)
        self.class_emb_dim = class_emb_dim
        self.d_model = api_emb_dim + class_emb_dim

    def forward(self, class_seq, api_seq):
        class_emb = self.class_emb_layer(class_seq)
        api_emb = self.api_emb_layer(api_seq)
        api_emb = torch.cat((class_emb, api_emb), dim=-1)
        return class_emb * math.sqrt(self.class_emb_dim), api_emb * math.sqrt(self.d_model)


def make_model(args, api_vocab_size, class_vocab_size):
    if args.model == 'lstm':
        model = StandardLSTM(api_vocab=api_vocab_size, api_emb_dim=args.api_emb_dim,
                             class_vocab=class_vocab_size, class_emb_dim=args.class_emb_dim,
                             hidden_size=args.hidden_size,
                             dropout=args.dropout)
    elif args.model == 'my_model':
        model = MyModel(api_vocab=api_vocab_size, api_emb_dim=args.api_emb_dim,
                        class_vocab=class_vocab_size, class_emb_dim=args.class_emb_dim,
                        hidden_size=args.hidden_size,
                        dropout=args.dropout,
                        class_attn=False)
    elif args.model == 'APIHelper':
        model = APIHelper(api_vocab=api_vocab_size, api_emb_dim=args.api_emb_dim,
                          class_vocab=class_vocab_size, class_emb_dim=args.class_emb_dim,
                          hidden_size=args.hidden_size,
                          dropout=args.dropout,
                          class_attn=False)
    elif args.model == 'APIHelper+':
        model = APIHelper(api_vocab=api_vocab_size, api_emb_dim=args.api_emb_dim,
                          class_vocab=class_vocab_size, class_emb_dim=args.class_emb_dim,
                          hidden_size=args.hidden_size,
                          dropout=args.dropout,
                          class_attn=True)
    elif args.model == 'my_model+':
        model = MyModel(api_vocab=api_vocab_size, api_emb_dim=args.api_emb_dim,
                        class_vocab=class_vocab_size, class_emb_dim=args.class_emb_dim,
                        hidden_size=args.hidden_size,
                        dropout=args.dropout,
                        class_attn=True)
    elif args.model == 'ngram':
        model = NGram(api_vocab=api_vocab_size, api_emb_dim=args.api_emb_dim,
                      n=4, dropout=args.dropout)
    return model
