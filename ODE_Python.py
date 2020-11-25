# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:12:28 2020

@author: Ahmed Osama Mahgoub
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


t = np.arange(0, 20, 0.1)
mu = 1
y01 = 2
y02 = [1, 1]

dydt1 = lambda y, t : 4 * np.exp(0.8*t) - 0.5*y

y1 = odeint(dydt1, y01, t)

fig1 = plt.figure()
ax1 = fig1.add_axes([0, 0, 1, 1])
ax1.plot(t, y1)
ax1.set_xlabel('t')
ax1.set_ylabel('Solution of ODE')
ax1.grid()

#####################################
# 2nd order ODE

dydt2 = lambda y, t, mu : [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]

y2= odeint(dydt2, y02, t, args = (mu,))

fig2 = plt.figure()
ax2 = fig2.add_axes([0, 0, 1, 1])
ax2.plot(t, y2[:,0], t, y2[:,1])
ax2.set_xlabel('t')
ax2.set_ylabel('Solution of ODE')
ax2.grid()