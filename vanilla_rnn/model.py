import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import nl_sys_gen_traj
from tqdm import tqdm

class RNNpred(nn.Module):
    def __init__(self, input_size, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_size):
        super(RNNpred, self).__init__()

        # Defining some parameters
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.hidden_dim_3 = hidden_dim_3
        self.input_size = input_size
        self.n_layers = 1

        # Defining the layers
        # RNN Layer
        self.rnn_1 = nn.RNN(input_size, hidden_dim_1)
        self.rnn_2 = nn.RNN(hidden_dim_1, hidden_dim_2)
        self.rnn_3 = nn.RNN(hidden_dim_2, hidden_dim_3)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim_3, output_size)

        # Initializing hidden state for first input using method defined below (Why should this be in the forward loop???)
        self.hidden_1, self.hidden_2, self_hidden_3 = self.init_hidden()

    def forward(self, x):
        # batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        # self.hidden_1, self.hidden_2, self.hidden_3 = self.init_hidden()

        # Passing in the input and hidden state into the model and obtaining outputs
        # out1, self.hidden_1 = self.rnn_1(x.view(-1, 1, self.input_size), self.hidden_1)
        out1, self.hidden_1 = self.rnn_1(x, self.hidden_1)
        out2, self.hidden_2 = self.rnn_2(out1, self.hidden_2)
        out3, self.hidden_3 = self.rnn_3(out2, self.hidden_3)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out2 = out2.contiguous().view(-1, self.hidden_dim_2)
        # pred = self.linear(lstm_out_2.view(len(seq),-1))
        out = self.fc(out3.view(len(x), -1))
        # out = self.fc(out2[:, -1, :])

        return out

    def init_hidden(self):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        # hidden_1 = torch.zeros(self.n_layers, batch_size, self.hidden_dim_1)
        # hidden_2 = torch.zeros(self.n_layers, batch_size, self.hidden_dim_2)
        # hidden_3 = torch.zeros(self.n_layers, batch_size, self.hidden_dim_3)
        hidden_1 = (torch.zeros(1, 1, self.hidden_dim_1), torch.zeros(1, 1, self.hidden_dim_1))
        hidden_2 = (torch.zeros(1, 1, self.hidden_dim_2), torch.zeros(1, 1, self.hidden_dim_2))
        hidden_3 = (torch.zeros(1, 1, self.hidden_dim_3), torch.zeros(1, 1, self.hidden_dim_3))
        # hidden_1 = torch.zeros(1, 1, self.hidden_dim_1)
        # hidden_2 = torch.zeros(1, 1, self.hidden_dim_2)
        # hidden_3 = torch.zeros(1, 1, self.hidden_dim_3)
        return hidden_1, hidden_2, hidden_3