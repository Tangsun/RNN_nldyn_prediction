import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import nl_sys_gen_traj
from model import RNNpred

model_save_name = 'RNN_500.pt'
model = RNNpred(input_size=1,hidden_dim_1=8, hidden_dim_2=7, hidden_dim_3 = 6,output_size=1)
model.load_state_dict(torch.load("trained_models/"+model_save_name))
# %%
batch_size = 500
num_batch = 2
horizon = batch_size

model.eval()

pred_model = list()
true_data = list()

train_traj_list = list()
pred_traj_list = list()
N = 500

# for i in range(1, 8):
#     x1_traj_i, _, _ = nl_sys_gen_traj('quad', i, N, 0.4)
#     x1_traj_i = x1_traj_i.reshape((-1, 1))
#     train_traj_list.append(x1_traj_i[: -1, :])
#     pred_traj_list.append(x1_traj_i[1:, :])
#
# for j in range(1, 4):
#     x1_traj_j, _, _ = nl_sys_gen_traj('henon', j, N, 0.5)
#     x1_traj_j = x1_traj_j.reshape((-1, 1))
#     train_traj_list.append(x1_traj_j[: -1, :])
#     pred_traj_list.append(x1_traj_j[1:, :])
#
# x1_traj_ikeda, _, _ = nl_sys_gen_traj('ikeda', 1, N, 0.5)
# x1_traj_ikeda = x1_traj_ikeda.reshape((-1, 1))
# train_traj_list.append(x1_traj_ikeda[:-1, :])
# pred_traj_list.append(x1_traj_ikeda[1:, :])

for k in range(1, 3):
    x1_traj_k, _, _ = nl_sys_gen_traj('sine', k, N, 0.5)
    x1_traj_k = x1_traj_k.reshape((-1, 1))
    train_traj_list.append(x1_traj_k[:-1, :])
    pred_traj_list.append(x1_traj_k[1:, :])

train_traj = np.concatenate(train_traj_list)
pred_traj = np.concatenate(pred_traj_list)
batch_size = N
num_batch  = len(train_traj_list)
#%%
train_set = torch.FloatTensor(train_traj).view(-1, 1)
true_set = torch.FloatTensor(pred_traj).view(-1, 1)

for id in range(num_batch):
    true_data.append(pred_traj_list[id])
    pred_traj = np.zeros((batch_size,))
    model.init_hidden(batch_size)

    for k in range(batch_size):
        pred_traj[k] = model(train_set[id * batch_size + k])

    pred_model.append(pred_traj)

# %%
fig4, ax4 = plt.subplots(1, 2, figsize=(12, 6))
T_index = ['20', '80']
for id in range(2):
    ax4[id].plot(range(batch_size), pred_model[id])
    ax4[id].plot(range(batch_size), true_data[id])
    ax4[id].set_xlim([200, 400])
    ax4[id].set_title('T=' + T_index[id])
plt.show()