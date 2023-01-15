import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import nl_sys_gen_traj, var_test_gen_traj
from model import LSTMmodel

model_save_name = 'LSTM_500_batch_lr_5.pt'
model = LSTMmodel(input_size=1,hidden_size_1=8, hidden_size_2=7, hidden_size_3 = 6,out_size=1)
model.load_state_dict(torch.load("../trained_models/"+model_save_name))
# %%
batch_size = 200
num_batch = 2
horizon = batch_size

model.eval()

pred_model = list()
true_data = list()

test_traj_list = list()
pred_traj_list = list()
N = 200

training_loss = list()

for k in range(1, 3):
    x1_traj_k, _, _ = var_test_gen_traj('sine', k, N, 0.5)
    x1_traj_k = x1_traj_k.reshape((-1, 1))
    test_traj_list.append(x1_traj_k[:-1])
    pred_traj_list.append(x1_traj_k[1:])

test_traj = np.concatenate(test_traj_list)
batch_size = N
num_batch  = len(test_traj_list)
#%%
test_set = torch.FloatTensor(test_traj).view(-1, 1)
criterion = nn.MSELoss()

for id in range(num_batch):
    gt = torch.tensor(pred_traj_list[id])
    true_data.append(gt)

    model.init_hidden()
    pred = model(test_set[id*batch_size: (id+1)*batch_size])
    # for k in range(batch_size):
    #     with torch.no_grad():
    #         # model.init_hidden()
    #         pred[k] = model(test_set[id * batch_size + k])

    pred_model.append(pred)
    training_loss.append(criterion(pred, gt))

# %%
fig4, ax4 = plt.subplots(1, 2, figsize=(12, 6))
T_index = ['20', '80']
for id in range(2):
    ax4[id].plot(range(batch_size), pred_model[id].detach())
    ax4[id].plot(range(batch_size), true_data[id])
    ax4[id].set_xlim([0, 200])
    ax4[id].set_title('T=' + T_index[id])
plt.show()
print(training_loss)