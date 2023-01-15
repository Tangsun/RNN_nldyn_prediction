import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

def CV_traj(q, X_0, T, dt):
    # q: parameter values
    # X_0: initial condition for states
    # T: Length of the trajectory
    phi_CV = np.array([[1, dt], [0, 1]])
    Phi = np.block([[phi_CV, np.zeros((2, 2))], [np.zeros((2, 2)), phi_CV]])

    q_CV = np.array([[dt**3/3, dt/2], [dt/2, 1]])
    q_CV = q*dt*q_CV
    Q = np.block([[q_CV, np.zeros((2, 2))], [np.zeros((2, 2)), q_CV]])

    X_i_traj = np.zeros((4, T+1))
    X_i_traj[:, 0] = X_0

    for k in range(T):
        xi_k = np.random.multivariate_normal([0., 0., 0., 0.], Q, 1)
        X_i_traj[:, k+1] = Phi.dot(X_i_traj[:, k]) + xi_k

    return X_i_traj

def CA_traj(q, X_0, T, dt):
    # q: parameter values
    # X_0: initial condition for states
    # T: Length of the trajectory
    phi_CA = np.array([[1, dt, dt**2/2], [0, 1, dt], [0, 0, 1]])
    Phi = np.block([[phi_CA, np.zeros((3, 3))], [np.zeros((3, 3)), phi_CA]])

    q_CA = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    q_CV = q * dt * q_CA
    Q = np.block([[q_CA, np.zeros((3, 3))], [np.zeros((3, 3)), q_CA]])

    X_i_traj = np.zeros((6, T + 1))
    X_i_traj[:, 0] = X_0

    for k in range(T):
        xi_k = np.random.multivariate_normal([0., 0., 0., 0., 0., 0.], Q, 1)
        X_i_traj[:, k + 1] = Phi.dot(X_i_traj[:, k]) + xi_k

    return X_i_traj

def TURN_traj(q, X_0, T, dt):
    # q: parameter values
    # X_0: initial condition for states
    # T: Length of the trajectory

    q_TURN = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    q_TURN = q * dt * q_TURN
    Q = np.block([[q_TURN, np.zeros((3, 3))], [np.zeros((3, 3)), q_TURN]])

    X_i_traj = np.zeros((6, T + 1))
    X_i_traj[:, 0] = X_0

    for k in range(T):
        v = np.sqrt(X_i_traj[1, k]**2 + X_i_traj[4, k]**2)
        a = np.sqrt(X_i_traj[2, k]**2 + X_i_traj[5, k]**2)
        omega = a/v
        phi_TURN = np.array([[1, np.sin(omega*dt)/omega, (1 - np.cos(omega*dt))/omega**2], [0, np.cos(omega*dt), np.sin(omega*dt)/omega], [0, -omega*np.sin(omega*dt), np.cos(omega*dt)]])
        Phi = np.block([[phi_TURN, np.zeros((3, 3))], [np.zeros((3, 3)), phi_TURN]])
        xi_k = np.random.multivariate_normal([0., 0., 0., 0., 0., 0.], Q, 1)
        X_i_traj[:, k + 1] = Phi.dot(X_i_traj[:, k]) + xi_k

    return X_i_traj