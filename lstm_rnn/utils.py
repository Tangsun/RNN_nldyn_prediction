import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

def nl_sys_gen_traj(type, i, N, x1_0):
    # only generate x_1 trajectory for each of the 13 time series
    x1_traj = np.zeros((N + 1,))
    x2_traj = np.zeros((N + 1,))
    m_traj = np.zeros((N + 1,))

    # x2_0 = 10*np.random.randn()
    x2_0 = 0
    x1_traj[0] = x1_0
    x2_traj[0] = x2_0

    alpha = np.array([1.5, 3.1, 3.6, 3.9, 3.95, 4.0])
    beta = np.array([0.8, 1.0])
    mu = 0.7
    T = np.array([20, 80])

    if type == 'quad':
        for k in range(N):
            # Note that the index of each type is starting from 1 (0 works but wrong)
            if i == 7:
                # alpha_k = (3.6+3.9)/2 + 0.15*np.sin(2*np.pi/5000*k)
                alpha_k = (3.6 + 3.9) / 2 + 0.15 * np.sin(2 * np.pi / N * k)
                x1_traj[k + 1] = alpha_k * x1_traj[k] * (1 - x1_traj[k])
            else:
                x1_traj[k + 1] = alpha[i - 1] * x1_traj[k] * (1 - x1_traj[k])
    elif type == 'henon':
        for k in range(N):
            if i == 3:
                # beta_k = 0.9 + 0.1*np.sin(2*np.pi/5000*k)
                beta_k = 0.9 + 0.1 * np.sin(2 * np.pi / N * k)
                x1_traj[k + 1] = beta_k - 1.4 * x1_traj[k] ** 2 + x2_traj[k]
                x2_traj[k + 1] = 0.3 * x1_traj[k]
            else:
                x1_traj[k + 1] = beta[i - 1] - 1.4 * x1_traj[k] ** 2 + x2_traj[k]
                x2_traj[k + 1] = 0.3 * x1_traj[k]
    elif type == 'ikeda':
        for k in range(N):
            m_traj[k] = 0.4 - 6.0 / (1 + x1_traj[k] ** 2 + x2_traj[k] ** 2)
            x1_traj[k + 1] = 1. + mu * (x1_traj[k] * np.cos(m_traj[k]) - x2_traj[k] * np.sin(m_traj[k]))
            x2_traj[k + 1] = mu * (x1_traj[k] * np.sin(m_traj[k]) + x2_traj[k] * np.cos(m_traj[k]))
        m_traj[N] = 0.4 - 6.0 / (1 + x1_traj[N] ** 2 + x2_traj[N] ** 2)
    elif type == 'sine':
        for k in range(N + 1):
            x1_traj[k] = np.sin(2 * np.pi * k / T[i - 1])
    else:
        print('Wrong type input')

    return x1_traj, x2_traj, m_traj


def var_test_gen_traj(type, i, N, x1_0):
    # only generate x_1 trajectory for each of the 13 time series
    x1_traj = np.zeros((N + 1,))
    x2_traj = np.zeros((N + 1,))
    m_traj = np.zeros((N + 1,))

    # x2_0 = 10*np.random.randn()
    x2_0 = 0
    x1_traj[0] = x1_0
    x2_traj[0] = x2_0

    alpha = np.array([3.0, 3.75])
    beta = np.array([0.75, 0.90])
    mu = 0.75
    T = np.array([15, 50, 85])

    if type == 'quad':
        for k in range(N):
            # Note that the index of each type is starting from 1 (0 works but wrong)
            x1_traj[k + 1] = alpha[i - 1] * x1_traj[k] * (1 - x1_traj[k])
    elif type == 'henon':
        for k in range(N):
            x1_traj[k + 1] = beta[i - 1] - 1.4 * x1_traj[k] ** 2 + x2_traj[k]
            x2_traj[k + 1] = 0.3 * x1_traj[k]
    elif type == 'ikeda':
        for k in range(N):
            m_traj[k] = 0.4 - 6.0 / (1 + x1_traj[k] ** 2 + x2_traj[k] ** 2)
            x1_traj[k + 1] = 1. + mu * (x1_traj[k] * np.cos(m_traj[k]) - x2_traj[k] * np.sin(m_traj[k]))
            x2_traj[k + 1] = mu * (x1_traj[k] * np.sin(m_traj[k]) + x2_traj[k] * np.cos(m_traj[k]))
        m_traj[N] = 0.4 - 6.0 / (1 + x1_traj[N] ** 2 + x2_traj[N] ** 2)
    elif type == 'sine':
        for k in range(N + 1):
            x1_traj[k] = np.sin(2 * np.pi * k / T[i - 1])
    else:
        print('Wrong type input')

    return x1_traj, x2_traj, m_traj

def switch_test_gen_traj(i, N1, N2, N3, x1_0):
    # only generate x_1 trajectory for each of the 13 time series
    N = N1 + N2 + N3
    x1_traj = np.zeros((N+1, ))
    x2_traj = np.zeros((N+1, ))
    m_traj = np.zeros((N+1, ))

    # x2_0 = 10*np.random.randn()
    x2_0 = 0
    x1_traj[0] = x1_0
    x2_traj[0] = x2_0

    mu = 0.7

    if i == 0:
        for k in range(N1):
            x1_traj[k+1] = 3.1*x1_traj[k]*(1 - x1_traj[k])
        for k in range(N1, N1+N2):
            x1_traj[k+1] = 0.8 - 1.4*x1_traj[k]**2 + x2_traj[k]
            x2_traj[k+1] = 0.3*x1_traj[k]
        for k in range(N1+N2, N1+N2+N3):
            x1_traj[k+1] = np.sin(2*np.pi*k/80)
    elif i == 1:
        for k in range(N1):
            x1_traj[k+1] = np.sin(2*np.pi*k/20)
        for k in range(N1, N1+N2+N3):
            x1_traj[k+1] = 3.1*x1_traj[k]*(1 - x1_traj[k])
    elif i == 2:
        for k in range(N1):
            x1_traj[k+1] = 0.8 - 1.4*x1_traj[k]**2 + x2_traj[k]
            x2_traj[k+1] = 0.3*x1_traj[k]
        for k in range(N1, N1+N2):
            x1_traj[k+1] = 1.0 - 1.4*x1_traj[k]**2 + x2_traj[k]
            x2_traj[k+1] = 0.3*x1_traj[k]
        for k in range(N1+N2, N1+N2+N3):
            x1_traj[k+1] = 3.1*x1_traj[k]*(1 - x1_traj[k])
    elif i == 3:
        for k in range(N1):
            x1_traj[k+1] = 1.5*x1_traj[k]*(1 - x1_traj[k])
        for k in range(N1, N1+N2):
            x1_traj[k+1] = 1.0 - 1.4*x1_traj[k]**2 + x2_traj[k]
            x2_traj[k+1] = 0.3*x1_traj[k]
        for k in range(N1+N2, N1+N2+N3):
            m_traj[k] = 0.4 - 6.0/(1 + x1_traj[k]**2 + x2_traj[k]**2)
            x1_traj[k+1] = 1. + mu*(x1_traj[k]*np.cos(m_traj[k]) - x2_traj[k]*np.sin(m_traj[k]))
            x2_traj[k+1] = mu*(x1_traj[k]*np.sin(m_traj[k]) + x2_traj[k]*np.cos(m_traj[k]))
        m_traj[N] = 0.4 - 6.0/(1 + x1_traj[N]**2 + x2_traj[N]**2)

    else:
        print('Wrong type input')

    return x1_traj, x2_traj, m_traj