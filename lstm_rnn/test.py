import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model import RNNpred
from utils import nl_sys_gen_traj
from tqdm import tqdm

model = RNNpred(input_size=1,hidden_dim_1=8, hidden_dim_2=7, hidden_dim_3 = 6,output_size=1)
model_save_name = 'RNN_500.pt'
torch.save(model.state_dict(), "trained_models/"+model_save_name)