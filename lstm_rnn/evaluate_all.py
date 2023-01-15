import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import nl_sys_gen_traj, var_test_gen_traj, switch_test_gen_traj
from model import LSTMmodel

model_save_name = 'final_LSTM_1000_rd_batch.pt'
model = LSTMmodel(input_size=1,hidden_size_1=8, hidden_size_2=7, hidden_size_3 = 6,out_size=1)
model.load_state_dict(torch.load("../trained_models/"+model_save_name))
# %%
batch_size = 5000
N = batch_size
# num_batch = 2
horizon = batch_size

model.eval()

# %%
test_T = batch_size

# Set the model to evaluation mode

model.eval()

x_init = 0.8
criterion = nn.MSELoss()

train_traj_list = list()
true_traj_list = list()
pred_traj_list = list()
train_loss = list()

for i in range(1, 8):
    x1_traj_i, _, _ = nl_sys_gen_traj('quad', i, N, 0.4)
    x1_traj_i = x1_traj_i.reshape((-1, 1))
    train_traj_list.append(x1_traj_i[: -1, :])
    true_traj_list.append(x1_traj_i[1:, :])

    x1_traj = torch.FloatTensor(x1_traj_i).view(-1, 1)
    model.init_hidden()
    x1_traj_pred = model(x1_traj[:-1])
    pred_traj_list.append(x1_traj_pred)
    train_loss.append(criterion(x1_traj_pred, torch.tensor(x1_traj_i[1:, :])))

for j in range(1, 4):
    x1_traj_j, _, _ = nl_sys_gen_traj('henon', j, N, 0.5)
    x1_traj_j = x1_traj_j.reshape((-1, 1))
    train_traj_list.append(x1_traj_j[: -1, :])
    true_traj_list.append(x1_traj_j[1:, :])

    x1_traj = torch.FloatTensor(x1_traj_j).view(-1, 1)
    model.init_hidden()
    x1_traj_pred = model(x1_traj[:-1])
    pred_traj_list.append(x1_traj_pred)
    train_loss.append(criterion(x1_traj_pred, torch.tensor(x1_traj_j[1:, :])))

x1_traj_ikeda, _, _ = nl_sys_gen_traj('ikeda', 1, N, 0.5)
x1_traj_ikeda = x1_traj_ikeda.reshape((-1, 1))
train_traj_list.append(x1_traj_ikeda[:-1, :])
true_traj_list.append(x1_traj_ikeda[1:, :])

x1_traj = torch.FloatTensor(x1_traj_ikeda).view(-1, 1)
model.init_hidden()
x1_traj_pred = model(x1_traj[:-1])
pred_traj_list.append(x1_traj_pred)
train_loss.append(criterion(x1_traj_pred, torch.tensor(x1_traj_ikeda[1:, :])))

for k in range(1, 3):
    x1_traj_k, _, _ = nl_sys_gen_traj('sine', k, N, 0.5)
    x1_traj_k = x1_traj_k.reshape((-1, 1))
    train_traj_list.append(x1_traj_k[:-1, :])
    true_traj_list.append(x1_traj_k[1:, :])

    x1_traj = torch.FloatTensor(x1_traj_k).view(-1, 1)
    model.init_hidden()
    x1_traj_pred = model(x1_traj[:-1])
    pred_traj_list.append(x1_traj_pred)
    train_loss.append(criterion(x1_traj_pred, torch.tensor(x1_traj_k[1:, :])))
# %%
fig1, ax1 = plt.subplots(2, 3, figsize=(18, 12))
alpha_index = ['3.1', '3.6', '3.9', '3.95', '4.0', 'periodic']
for id in range(1, 7):
    ax1[(id - 1) // 3, (id - 1) % 3].plot(range(batch_size), pred_traj_list[id].detach())
    ax1[(id - 1) // 3, (id - 1) % 3].plot(range(batch_size), true_traj_list[id])
    ax1[(id - 1) // 3, (id - 1) % 3].set_xlim([2650, 2750])
    ax1[(id - 1) // 3, (id - 1) % 3].set_title('alpha=' + alpha_index[id - 1])
# %%
fig2, ax2 = plt.subplots(1, 3, figsize=(18, 6))
beta_index = ['0.8', '1.0', 'periodic']
for id in range(7, 10):
    ax2[(id - 7) % 3].plot(range(batch_size), pred_traj_list[id].detach())
    ax2[(id - 7) % 3].plot(range(batch_size), true_traj_list[id])
    ax2[(id - 7) % 3].set_xlim([2650, 2750])
    ax2[(id - 7) % 3].set_title('beta=' + beta_index[id - 7])
# %%
fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
ax3.plot(range(batch_size), pred_traj_list[10].detach())
ax3.plot(range(batch_size), true_traj_list[10])
ax3.set_xlim([2650, 2750])
ax3.set_title('ikeda')
# %%
fig4, ax4 = plt.subplots(1, 2, figsize=(12, 6))
T_index = ['20', '80']
for id in range(11, 13):
    ax4[(id - 11) % 2].plot(range(batch_size), pred_traj_list[id].detach())
    ax4[(id - 11) % 2].plot(range(batch_size), true_traj_list[id])
    ax4[(id - 11) % 2].set_xlim([2650, 2750])
    ax4[(id - 11) % 2].set_title('T=' + T_index[id - 11])

test_var_input = list()
test_var_pred = list()
test_var_out = list()
test_loss = list()

for i in range(2):
    x1_traj_pred = np.zeros((test_T,))
    x1_traj, _, _ = var_test_gen_traj('quad', i, test_T, x_init)
    test_var_input.append(x1_traj[:-1])
    test_var_out.append(x1_traj[1:])

    x1_traj = torch.FloatTensor(x1_traj).view(-1, 1)
    model.init_hidden()
    x1_traj_pred = model(x1_traj[:-1])
    test_var_pred.append(x1_traj_pred)
    test_loss.append(criterion(x1_traj_pred, torch.tensor(x1_traj[1:])))

fig5, ax5 = plt.subplots(1, 2, figsize=(12, 6))
alpha_index = ['3.0', '3.75']

for id in range(2):
    ax5[id].plot(range(test_T), test_var_pred[id].detach())
    ax5[id].plot(range(test_T), test_var_out[id])
    ax5[id].set_xlim([2650, 2750])
    ax5[id].set_title('alpha=' + alpha_index[id])

for i in range(2):
    x1_traj_pred = np.zeros((test_T,))
    x1_traj, _, _ = var_test_gen_traj('henon', i, test_T, x_init)
    test_var_input.append(x1_traj[:-1])
    test_var_out.append(x1_traj[1:])

    x1_traj = torch.FloatTensor(x1_traj).view(-1, 1)
    model.init_hidden()
    x1_traj_pred = model(x1_traj[:-1])
    test_var_pred.append(x1_traj_pred)
    test_loss.append(criterion(x1_traj_pred, torch.tensor(x1_traj[1:])))

fig6, ax6 = plt.subplots(1, 2, figsize=(12, 6))
beta_index = ['0.75', '0.90']
for id in range(2):
    ax6[id].plot(range(test_T), test_var_pred[id + 2].detach())
    ax6[id].plot(range(test_T), test_var_out[id + 2])
    ax6[id].set_xlim([2650, 2750])
    ax6[id].set_title('beta=' + beta_index[id])
# %%
x1_traj_pred = np.zeros((test_T,))
x1_traj, _, _ = var_test_gen_traj('ikeda', 0, test_T, x_init)
test_var_input.append(x1_traj[:-1])
test_var_out.append(x1_traj[1:])

x1_traj = torch.FloatTensor(x1_traj).view(-1, 1)
model.init_hidden()
x1_traj_pred = model(x1_traj[:-1])
test_var_pred.append(x1_traj_pred)
test_loss.append(criterion(x1_traj_pred, torch.tensor(x1_traj[1:])))

fig7, ax7 = plt.subplots(1, 1, figsize=(6, 6))
ikeda_index = '0.75'
ax7.plot(range(test_T), test_var_pred[4].detach())
ax7.plot(range(test_T), test_var_out[4])
ax7.set_xlim([2650, 2750])
ax7.set_title('mu=' + ikeda_index)
# %%
for i in range(3):
    x1_traj_pred = np.zeros((test_T,))
    x1_traj, _, _ = var_test_gen_traj('sine', i, test_T, x_init)
    test_var_input.append(x1_traj[:-1])
    test_var_out.append(x1_traj[1:])

    x1_traj = torch.FloatTensor(x1_traj).view(-1, 1)
    model.init_hidden()
    x1_traj_pred = model(x1_traj[:-1])
    test_var_pred.append(x1_traj_pred)
    test_loss.append(criterion(x1_traj_pred, torch.tensor(x1_traj[1:])))

fig8, ax8 = plt.subplots(1, 3, figsize=(18, 6))
T_index = ['15', '50', '85']
for id in range(3):
    ax8[id].plot(range(test_T), test_var_pred[id + 5].detach())
    ax8[id].plot(range(test_T), test_var_out[id + 5])
    ax8[id].set_xlim([2650, 2750])
    ax8[id].set_title('T=' + T_index[id])

test_sw_input = list()
test_sw_pred = list()
test_sw_out = list()
test_sw_loss = list()

test_T1 = 100
test_T2 = 200
test_T3 = 300
test_T = test_T1 + test_T2 + test_T3
x_init = 0.6

for i in range(4):
    x1_traj_pred = np.zeros((test_T,))
    x1_traj, _, _ = switch_test_gen_traj(i, test_T1, test_T2, test_T3, x_init)
    test_sw_input.append(x1_traj[:-1])
    test_sw_out.append(x1_traj[1:])

    x1_traj = torch.FloatTensor(x1_traj).view(-1, 1)
    model.init_hidden()
    x1_traj_pred = model(x1_traj[:-1])
    test_sw_pred.append(x1_traj_pred)
    test_sw_loss.append(criterion(x1_traj_pred, torch.tensor(x1_traj[1:])))

fig9, ax9 = plt.subplots(1, 2, figsize=(12, 6))
ind = [0, 3]
for id in [0, 1]:
    ax9[id].plot(range(test_T), test_sw_pred[ind[id]].detach())
    ax9[id].plot(range(test_T), test_sw_out[ind[id]])
    # if id != 1:
    ax9[id].set_xlim([75, 150])

fig10, ax10 = plt.subplots(1, 2, figsize=(12, 6))
for id in [0, 1]:
    ax10[id].plot(range(test_T), test_sw_pred[ind[id]].detach())
    ax10[id].plot(range(test_T), test_sw_out[ind[id]])
    # if id != 1:
    ax10[id].set_xlim([290, 360])

plt.show()
print('Training loss:', train_loss)
print('Testing loss:', test_loss)
print('Switch Test Loss:', test_sw_loss)