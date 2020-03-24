import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StandardLSTM(nn.Module):
    def __init__(self, api_vocab, api_emb_dim,
                 class_vocab, class_emb_dim,
                 hidden_size,
                 dropout=0.0):
        super(StandardLSTM, self).__init__()
        self.emb_layer = APIEmbLayer(api_vocab, api_emb_dim, class_vocab, class_emb_dim)
        self.rnn = nn.LSTM(input_size=api_emb_dim+class_emb_dim,
                           hidden_size=hidden_size)
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=api_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        class_seq, api_seq = inputs
        api_len = torch.sum(api_seq != 0, dim=1)
        api_len = (api_len - 1).tolist()
        api_emb = self.emb_layer(class_seq, api_seq)
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

        self.d_model = api_emb_dim + class_emb_dim

    def forward(self, class_seq, api_seq):
        class_emb = self.class_emb_layer(class_seq)
        api_emb = self.api_emb_layer(api_seq)
        api_emb = torch.cat((class_emb, api_emb), dim=-1)
        return api_emb * math.sqrt(self.d_model)


def make_model(args, api_vocab_size, class_vocab_size):
    model = StandardLSTM(api_vocab=api_vocab_size, api_emb_dim=args.api_emb_dim,
                         class_vocab=class_vocab_size, class_emb_dim=args.class_emb_dim,
                         hidden_size=args.hidden_size,
                         dropout=args.dropout)
    return model
