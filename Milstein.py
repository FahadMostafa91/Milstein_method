#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:41:06 2023

@author: Fahad
"""

# -*- coding: utf-8 -*-
# Milstein Method

import numpy as np
import matplotlib.pyplot as plt

num_sims = 1  # One Example

# five Second and thousand grid points
t_init, t_end = 0, 1
N = 10000 # Compute 10000 grid points
dt = float(t_end - t_init) / N

## Initial Conditions
y_init = 1
μ, σ = 5, 1.5  # mean and variance

# dw Random process, weiner process 
def dW(delta_t):
    """Random sample gaussian distribution"""
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

# vectors in terms of data 
ts = np.arange(t_init, t_end + dt, dt)
ys = np.zeros(N + 1)
ys[0] = y_init

# Loop
for _ in range(num_sims):
    for i in range(1, ts.size):
        t = (i - 1) * dt
        y = ys[i - 1]
        # Milstein method
        dw_ = dW(dt)
        ys[i] = y + μ * dt * y + σ * y * dw_ + 0.5 * σ**2 * y * (dw_**2 - dt)
    plt.plot(ts, ys)

# Plot
plt.xlabel("time (second)")
plt.grid()
h = plt.ylabel("Y ")
h.set_rotation(0)
plt.show()