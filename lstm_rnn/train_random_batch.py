import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model import LSTMmodel
from utils import nl_sys_gen_traj
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

x1_traj, _, _ = nl_sys_gen_traj('sine', 1, 100, 1)

# fig, ax = plt.subplots()
# ax.plot(range(101), x1_traj)
# plt.xlim([0, 101])
# plt.ylim([-1., 1.])
# plt.show()

train_traj_list = list()
pred_traj_list = list()
N = 5000

for i in range(1, 8):
    x1_traj_i, _, _ = nl_sys_gen_traj('quad', i, N, 0.4)
    x1_traj_i = x1_traj_i.reshape((-1, 1))
    train_traj_list.append(x1_traj_i[: -1, :])
    pred_traj_list.append(x1_traj_i[1:, :])

for j in range(1, 4):
    x1_traj_j, _, _ = nl_sys_gen_traj('henon', j, N, 0.5)
    x1_traj_j = x1_traj_j.reshape((-1, 1))
    train_traj_list.append(x1_traj_j[: -1, :])
    pred_traj_list.append(x1_traj_j[1:, :])

x1_traj_ikeda, _, _ = nl_sys_gen_traj('ikeda', 1, N, 0.5)
x1_traj_ikeda = x1_traj_ikeda.reshape((-1, 1))
train_traj_list.append(x1_traj_ikeda[:-1, :])
pred_traj_list.append(x1_traj_ikeda[1:, :])

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

model = LSTMmodel(input_size=1,hidden_size_1=8, hidden_size_2=7, hidden_size_3 = 6,out_size=1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.005)

# epochs = 200
epochs = 1000

count = 0
for epoch in tqdm(range(epochs)):

    # Running each batch separately
    batch_seq = random.sample(range(num_batch), num_batch)

    for bat_id in batch_seq:

        # set the optimization gradient to zero

        optimizer.zero_grad()

        # Make predictions on the current sequence
        model.init_hidden()

        y_pred = model(train_set[bat_id * batch_size: (bat_id+1)*batch_size])

        # Compute the loss

        loss = criterion(y_pred, true_set[bat_id * batch_size: (bat_id+1)*batch_size])
        writer.add_scalar("Loss/Train", loss, count)
        count = count+1

        # Perform back propogation and gradient descent

        loss.backward()

        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch: {epoch} Loss: {loss.item():10.8f}')

writer.flush()
writer.close()
model_save_name = 'final_LSTM_1000_rd_batch.pt'
torch.save(model.state_dict(), "trained_models/"+model_save_name)