import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import CV_traj, CA_traj, TURN_traj
from model import LSTMmodel

model_save_name = 'LSTM_200_lr_5.pt'
model = LSTMmodel(input_size=1,hidden_size_1=8, hidden_size_2=7, hidden_size_3 = 6,out_size=1)
model.load_state_dict(torch.load("trained_models/"+model_save_name))
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

X_0_CV = np.array([0., 10., 0., 0.])
q_list = np.array([1., 4., 8., 10.])
dt = 0.1
for i in range(4):
    X_i_traj = CV_traj(q_list[i], X_0_CV, N, dt)
    test_traj_list.append(X_i_traj[:, :-1])
    pred_traj_list.append(X_i_traj[:, 1:])

test_traj = np.concatenate(test_traj_list)
test_set = torch.FloatTensor(test_traj).view(-1, 1)

criterion = nn.MSELoss()
for id in range(16):
    gt = torch.tensor(pred_traj_list[id//4][id%4, :])
    true_data.append(gt)

    model.init_hidden()
    pred = model(test_set[id* N: (id+1)*N])
    # for k in range(batch_size):
    #     with torch.no_grad():
    #         # model.init_hidden()
    #         pred[k] = model(test_set[id * batch_size + k])

    pred_model.append(pred)
    training_loss.append(criterion(pred, gt))

# %%
fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6))
q_index = ['1', '4']
ax1[0].plot(range(N), pred_model[0].detach())
ax1[0].plot(range(N), true_data[0])
ax1[0].set_xlabel('step k')
ax1[0].set_ylabel('x_k')
ax1[0].set_xlim([0, 200])
ax1[0].set_title('q=' + q_index[0])
ax1[1].plot(range(N), pred_model[2].detach())
ax1[1].plot(range(N), true_data[2])
ax1[1].set_xlim([0, 200])
ax1[1].set_xlabel('step k')
ax1[1].set_ylabel('y_k')
ax1[1].set_title('q=' + q_index[0])
plt.show()
print(training_loss)