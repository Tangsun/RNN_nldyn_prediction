import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# from model import RNNpred
from utils import CV_traj, CA_traj, TURN_traj
from tqdm import tqdm

# a = np.random.multivariate_normal([0., 0., 0., 0.], np.identity(4), 1).transpose()
# print(a)
# X_0 = np.array([0., 10., 0., 0.])
#
# fig1, ax1 = plt.subplots(4, 1, figsize=(6, 24))
# X = CV_traj(5, X_0, 200, 1)
# for id in range(4):
#     ax1[id].plot(range(201), X[id, :])
#
# X_0 = np.array([0., 10., 1., 0., 0., 1.])
# X_CA = CA_traj(5, X_0, 200, 1)
# fig2, ax2 = plt.subplots(6, 1, figsize=(6, 36))
# for id in range(6):
#     ax2[id].plot(range(201), X_CA[id, :])

X_0 = np.array([0., 10., 1., 0., 0., 1.])
X_TURN = TURN_traj(1, X_0, 200, 1)
fig2, ax2 = plt.subplots(7, 1, figsize=(6, 42))
for id in range(7):
    if id < 6:
        ax2[id].plot(range(201), X_TURN[id, :])
    else:
        ax2[id].plot(X_TURN[0, :], X_TURN[3, :])


plt.show()