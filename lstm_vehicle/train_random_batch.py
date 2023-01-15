import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model import LSTMmodel
from utils import CV_traj, CA_traj, TURN_traj
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

train_traj_list = list()
pred_traj_list = list()
N = 200
dt = 0.1

X_0_CV = np.array([0., 10., 0., 0.])
q_list = np.array([1., 4., 8., 10.])
for i in range(4):
    X_i_traj = CV_traj(q_list[i], X_0_CV, N, dt)
    train_traj_list.append(X_i_traj[:, :-1])
    pred_traj_list.append(X_i_traj[:, 1:])

train_traj = np.concatenate(train_traj_list)
pred_traj = np.concatenate(pred_traj_list)
batch_size = N
num_batch  = len(train_traj_list)*4
#%%
train_set = torch.FloatTensor(train_traj).view(-1, 1)
true_set = torch.FloatTensor(pred_traj).view(-1, 1)

model = LSTMmodel(input_size=1,hidden_size_1=8, hidden_size_2=7, hidden_size_3 = 6,out_size=1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.005)

# epochs = 200
epochs = 200

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
model_save_name = 'LSTM_200_lr_5.pt'
torch.save(model.state_dict(), "trained_models/"+model_save_name)