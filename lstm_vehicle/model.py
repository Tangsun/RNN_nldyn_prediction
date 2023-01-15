import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm


class LSTMmodel(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, out_size):
        super().__init__()

        self.hidden_size_1 = hidden_size_1

        self.hidden_size_2 = hidden_size_2

        self.hidden_size_3 = hidden_size_3

        self.input_size = input_size

        self.lstm_1 = nn.LSTM(input_size, hidden_size_1)

        self.lstm_2 = nn.LSTM(hidden_size_1, hidden_size_2)

        self.lstm_3 = nn.LSTM(hidden_size_2, hidden_size_3)

        self.linear = nn.Linear(hidden_size_3, out_size)

        self.hidden_1 = (torch.zeros(1, 1, hidden_size_1), torch.zeros(1, 1, hidden_size_1))

        self.hidden_2 = (torch.zeros(1, 1, hidden_size_2), torch.zeros(1, 1, hidden_size_2))

        self.hidden_3 = (torch.zeros(1, 1, hidden_size_3), torch.zeros(1, 1, hidden_size_3))

    def forward(self, seq):
        lstm_out_1, self.hidden_1 = self.lstm_1(seq.view(-1, 1, self.input_size), self.hidden_1)

        lstm_out_2, self.hidden_2 = self.lstm_2(lstm_out_1, self.hidden_2)

        lstm_out_3, self.hidden_3 = self.lstm_3(lstm_out_2, self.hidden_3)

        pred = self.linear(lstm_out_3.view(len(seq), -1))

        return pred

    def init_hidden(self):
        self.hidden_1 = (torch.zeros(1, 1, self.hidden_size_1),
                          torch.zeros(1, 1, self.hidden_size_1))

        self.hidden_2 = (torch.zeros(1, 1, self.hidden_size_2),
                          torch.zeros(1, 1, self.hidden_size_2))

        self.hidden_3 = (torch.zeros(1, 1, self.hidden_size_3),
                          torch.zeros(1, 1, self.hidden_size_3))