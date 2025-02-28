import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

class PLACEHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        # self.transform = BertPredictionHeadTransform(config)
         # self.transform = BertPredictionHeadTransform(config)
        self.ac = nn.ReLU()
        self.fc = nn.Linear(config.hidden_size, 1)
        # self.fc2 = nn.Linear(256, 1, bias=True)
        # self.batchnorm = nn.BatchNorm1d()
        # self.decoder_2 = nn.Linear(64, 1, bias=True)
        # self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # if weight is not None:
        #     self.fc1.weight = weight

    def forward(self, x):
        # x = self.transform(x)
        x_ = self.ac(x)
        out = self.fc(x_)
        # x = self.ac1(x)
        # x = self.fc2(x)
        # x = self.batchnorm(x)
        # x = self.decoder_2(x)
        return out

class MPPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 256 * 3)

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x
