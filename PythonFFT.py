# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 13:29:19 2021
White noise, FFT, and PSD
@author: Ahmed Osama Mahgoub - UNSW - 2021
"""

import numpy as np
import numpy.fft as npfft
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random
import pandas as pd

#%% White noise signal generation
# Seed random number generator
# random.seed(16785)

# White noise series generation
WNsignal = [random.gauss(0.0,1.0) for i in range(1000)]
WNsignal = pd.Series(WNsignal)

# Plotting the signal
fig1 = plt.figure()
ax1 = fig1.add_axes([0,0,1,1])
ax1.plot(WNsignal)
ax1.set_xlabel('index')
ax1.set_ylabel('signal')
ax1.set_title('Random white noise signal')
ax1.grid(True)

# Histogram for the white noise signal
WNsignal.describe()
fig2 = plt.figure()
WNsignal.hist(bins = 30, grid = False)

# Autocorrelation plot
pd.plotting.autocorrelation_plot(WNsignal)

#%% FFT and PSD
np.random.seed(19695601)
diff = 0.01
ax = np.arange(0, 10, diff)
n = np.random.randn(len(ax))
by = np.exp(-ax/0.05)

cn = np.convolve(n, by)
cn = cn[:len(ax)]
s = 0.1 * np.sin(2 *np.pi *ax) + cn

freq_sample = 0.5 * (1/diff) # Sampling frequency
df = 1/ax[-1]
freq = np.arange(df, freq_sample, df)
signal_fft = npfft.fft(s)
signal_fft = signal_fft[:len(freq)]

fig0 = plt.figure()
ax0 = fig0.add_axes([0,0,1,1])
ax0.plot(freq, signal_fft)
ax0.set_xlabel('Frequency (Hz)')
ax0.set_ylabel('FFT')
ax0.set_title('FFT for the random signal')
ax0.grid(True)

fig3, (ax31, ax32) = plt.subplots(2,1)

ax31.plot(ax, s)
ax31.set_xlabel('time')
ax31.set_ylabel('signal')
ax31.set_title('Generated signal for PSD')
ax31.grid(True)

ax32.psd(s, 512, 1/diff)
ax32.set_xscale('log')

#%% Second example
random_rep = np.random.RandomState(19680801)  

# Signal
fps = 1000
a = np.linspace(0, 0.3, 301)
b = np.array([2, 8]).reshape(-1,1)
c = np.array([150, 140]).reshape(-1,1)
d = (b * np.exp(2j * np.pi * c * a)).sum(axis = 0) + 5 * random_rep.randn(*a.shape)

fig4, (ax41, ax42) = plt.subplots(ncols=2, constrained_layout = True)

# Variables to define plotting range and ticks locations
e = np.arange(-50, 30, 10)
f = (e[0], e[-1])
g = np.arange(-500, 550, 200)

ax41.psd(d, NFFT = 301,
         Fs = fps,
         window = mlab.window_none,
         pad_to = 1024,
         scale_by_freq = True)
ax41.set_title('Periodo-gram')
ax41.set_yticks(e)
ax41.set_xticks(g)
ax41.grid(True)
ax41.set_ylim(f)

ax42.psd(d, NFFT = 150,
         Fs = fps,
         window = mlab.window_none,
         pad_to = 512,
         noverlap = 75,
         scale_by_freq = True)
ax42.set_title('Welch')
ax42.set_ylabel('')
ax42.set_yticks(e)
ax42.set_xticks(g)
ax42.grid(True)
ax42.set_ylim(f)
