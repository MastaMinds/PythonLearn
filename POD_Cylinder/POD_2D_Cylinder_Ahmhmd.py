# -*- coding: utf-8 -*-
"""
Python code for obtaining the POD modes for a 2D laminar flow
past a Cylinder

Made by Ahmhmd UNSW - 2021
"""

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d
import pandas as pd
from scipy.interpolate import griddata

Main_path = 'C:/Users/ahmhm/POD_Cylinder/Cylinder_2D_Laminar/'

dt = 0.1
T_final = 100
t = np.arange(dt, T_final +dt, dt)
N = 900
Main_file = Main_path + 'Cylinder_2D_Foam_1.csv'
filenames = N*[0]
data_all = N*[0]

test_table = pd.read_csv(Main_file)
column_names = test_table.columns
n_nodes = test_table.shape[0]

# Data matrix
X = np.zeros([n_nodes, N])
for i in range(N):
    filenames[i] = Main_path + 'Cylinder_2D_Foam_' + str(i+1) + '.csv'
    data_all[i] = pd.read_csv(filenames[i])
    
    # Append data matrix using current snapshot data
    X[:,i] = data_all[i]['U:0']
    # X[:,i] = data_all[i]['vorticity:2']
    print('Finished data file number'+str(i+1))

X_mean = X.mean(axis=1)
for k in range(N):
    X[:,k] = X[:,k] - X_mean
    
U, Sigma, V = np.linalg.svd(X)

y = test_table['Points:0']
x = test_table['Points:1']
z = test_table['vorticity:2']

y_a, x_a = np.mgrid[-0.2:1:0.0005, -0.5:0.5:0.0005]
z_a = griddata((x,y), z, (x_a, y_a), method='linear')
U0 = griddata((x,y), U[:,0], (x_a, y_a), method='linear')
U1 = griddata((x,y), U[:,1], (x_a, y_a), method='linear')
U2 = griddata((x,y), U[:,2], (x_a, y_a), method='linear')
U3 = griddata((x,y), U[:,3], (x_a, y_a), method='linear')
U4 = griddata((x,y), U[:,4], (x_a, y_a), method='linear')

x_a[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
y_a[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
z_a[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
U0[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
U1[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
U2[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
U3[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
U4[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan


ax00 = plt.axes(projection='3d')
ax00.view_init(-90, 0)
ax00.plot_surface(x_a, y_a, z_a, cmap= 'Spectral',linewidth=0, 
                  vmin = -2, vmax = 2,antialiased=False)
ax00.set_ylim(-0.1, 1)
ax00.set_xlim(-0.5, 0.5)

ax01 = plt.axes(projection='3d')
ax01.view_init(-90, 0)
ax01.plot_surface(x_a, y_a, U2, cmap= 'Spectral',linewidth=0,
                  antialiased=False)
ax01.set_ylim(-0.1, 1)
ax01.set_xlim(-0.5, 0.5)

TriGrid = mp.tri.Triangulation(x, y)

fig = plt.figure()
ax = mplot3d.Axes3D(fig)
ax.view_init(-90, 0)
surf = ax.plot_trisurf(x, y, z, triangles = TriGrid.triangles, vmin = -2,
                       vmax = 2, cmap = 'Spectral', edgecolor = 'none',
                       linewidth = 0, antialiased = False) # vmax and vmin: 'None'
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(-0.5, 0.5)
plt.colorbar(surf)

fig0 = plt.figure()
ax0 = mplot3d.Axes3D(fig0)
ax0.view_init(-90, 0)
surf0 = ax0.plot_trisurf(x, y, U[:,0], triangles = TriGrid.triangles, vmin = None,
                       vmax = None, cmap = 'Spectral', edgecolor = 'none',
                       linewidth = 0, antialiased = False) # vmax and vmin: 'None'
ax0.set_ylim(0, 1)
ax0.set_xlim(-0.5, 0.5)
plt.colorbar(surf0)

fig1 = plt.figure()
ax1 = mplot3d.Axes3D(fig1)
ax1.view_init(-90, 0)
surf1 = ax1.plot_trisurf(x, y, U[:,1], triangles = TriGrid.triangles, vmin = None,
                       vmax = None, cmap = 'Spectral', edgecolor = 'none',
                       linewidth = 0, antialiased = False) # vmax and vmin: 'None'
ax1.set_ylim(0, 1)
ax1.set_xlim(-0.5, 0.5)
plt.colorbar(surf1)

fig2 = plt.figure()
ax2 = mplot3d.Axes3D(fig2)
ax2.view_init(-90, 0)
surf2 = ax2.plot_trisurf(x, y, U[:,2], triangles = TriGrid.triangles, vmin = None,
                       vmax = None, cmap = 'Spectral', edgecolor = 'none',
                       linewidth = 0, antialiased = False) # vmax and vmin: 'None'
ax2.set_ylim(0, 1)
ax2.set_xlim(-0.5, 0.5)
plt.colorbar(surf2)

fig3 = plt.figure()
ax3 = mplot3d.Axes3D(fig3)
ax3.view_init(-90, 0)
surf3 = ax3.plot_trisurf(x, y, U[:,3], triangles = TriGrid.triangles, vmin = None,
                       vmax = None, cmap = 'Spectral', edgecolor = 'none',
                       linewidth = 0, antialiased = False) # vmax and vmin: 'None'
ax3.set_ylim(0, 1)
ax3.set_xlim(-0.5, 0.5)
plt.colorbar(surf3)

fig4 = plt.figure()
ax4 = mplot3d.Axes3D(fig4)
ax4.view_init(-90, 0)
surf4 = ax4.plot_trisurf(x, y, U[:,4], triangles = TriGrid.triangles, vmin = None,
                       vmax = None, cmap = 'Spectral', edgecolor = 'none',
                       linewidth = 0, antialiased = False) # vmax and vmin: 'None'
ax4.set_ylim(0, 1)
ax4.set_xlim(-0.5, 0.5)
plt.colorbar(surf4)