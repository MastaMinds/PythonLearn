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
from AhmhmdFunctions import *

# Main_path = 'C:/Users/ahmhm/POD_Cylinder/Cylinder_2D_Laminar/'
Main_path = 'C:/Users/ahmhm/POD_Cylinder/New_Cylinder_Case/'

dt = 2
T_final = 400
t = np.arange(dt, T_final +dt, dt)
N = 200
Main_file = Main_path + 'Cylinder_2D_Foam_1.csv'
filenames = N*[0]
data_all = N*[0]

test_table = pd.read_csv(Main_file)
column_names = test_table.columns
n_nodes = test_table.shape[0]

# Data matrix
X = np.zeros([n_nodes, N])
for i in range(N):
    filenames[i] = Main_path + 'Cylinder_2D_Foam_' + str(i) + '.csv'
    data_all[i] = pd.read_csv(filenames[i])
    
    # Append data matrix using current snapshot data
    X[:,i] = data_all[i]['U:0']
    # X[:,i] = data_all[i]['vorticity:2']
    print('Finished data file number'+str(i+1))

X_mean = X.mean(axis=1)
for k in range(N):
    X[:,k] = X[:,k] - X_mean
    
U, Sigma, V = np.linalg.svd(X)

E_tot = np.zeros((N,))
for i in range(len(Sigma)):
    E_tot[i] = Sigma[:i+1].sum()

x = test_table['Points:0']
y = test_table['Points:1']
z = test_table['vorticity:2']

x0 = -0.2
x_end = 1
dx = 0.0005
y0 = -0.5
y_end = 0.5
dy = 0.0005
x_a1 = np.arange(x0, x_end, dx)
y_a1 = np.arange(y0, y_end, dy)
x_ar = [x0, x_end, dx]
y_ar = [y0, y_end, dy]

x_a, y_a, z_a = get2DGridData(x, y, z,  x_ar, y_ar)
_, _, U0 = get2DGridData(x, y, U[:,0],  x_ar, y_ar)
_, _, U1 = get2DGridData(x, y, U[:,1],  x_ar, y_ar)
_, _, U2 = get2DGridData(x, y, U[:,2],  x_ar, y_ar)
_, _, U3 = get2DGridData(x, y, U[:,3],  x_ar, y_ar)
_, _, U4 = get2DGridData(x, y, U[:,4],  x_ar, y_ar)
_, _, U5 = get2DGridData(x, y, U[:,5],  x_ar, y_ar)
_, _, U6 = get2DGridData(x, y, U[:,6],  x_ar, y_ar)
_, _, U7 = get2DGridData(x, y, U[:,7],  x_ar, y_ar)

# x_a[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
# y_a[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
z_a[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
U0[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
U1[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
U2[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
U3[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
U4[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
U5[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
U6[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan
U7[np.sqrt(x_a**2 + y_a**2) <= 0.1] = np.nan

#%% Plotting results

ax00 = plt.axes(projection='3d')
ax00.view_init(-90, 0)
ax00.plot_surface(x_a, y_a, z_a, cmap= 'Spectral',linewidth=0, 
                  vmin = -2, vmax = 2,antialiased=False)
ax00.set_ylim(-0.1, 1)
ax00.set_xlim(-0.5, 0.5)


TriGrid = mp.tri.Triangulation(x, y)

fig0 = plt.figure()
ax0 = mplot3d.Axes3D(fig0)
ax0.view_init(-90, 0)
surf0 = ax0.plot_trisurf(x, y, U[:,0], triangles = TriGrid.triangles, vmin = None,
                       vmax = None, cmap = 'Spectral', edgecolor = 'none',
                       linewidth = 0, antialiased = False) # vmax and vmin: 'None'
ax0.set_ylim(0, 1)
ax0.set_xlim(-0.5, 0.5)
fig0.colorbar(surf0)

fig8 = plt.figure()
ax8 = fig8.add_axes([0,0,1,1])
surf8 = ax8.pcolor(x_a, y_a, U0, cmap= 'seismic' , shading = 'auto')
fig8.colorbar(surf8)
ax8.set_ylim(-0.5, 0.5)
ax8.set_xlim(-0.2, 1)

fig6 = plt.figure()
ax6 = fig6.add_axes([0,0,1,1])
surf6 = ax6.pcolor(x_a, y_a, U1, cmap= 'seismic' , shading = 'auto')
fig6.colorbar(surf6)
ax6.set_ylim(-0.5, 0.5)
ax6.set_xlim(-0.2, 1)

fig5 = plt.figure()
ax5 = fig5.add_axes([0,0,1,1])
surf5 = ax5.pcolor(x_a, y_a, U2, cmap= 'seismic' , shading = 'auto')
fig5.colorbar(surf5)
ax5.set_ylim(-0.5, 0.5)
ax5.set_xlim(-0.2, 1)

fig7 = plt.figure()
ax7 = fig7.add_axes([0,0,1,1])
surf7 = ax7.pcolor(x_a, y_a, U3, cmap= 'seismic' , shading = 'auto')
fig7.colorbar(surf7)
ax7.set_ylim(-0.5, 0.5)
ax7.set_xlim(-0.2, 1)

fig9 = plt.figure()
ax9 = fig9.add_axes([0,0,1,1])
surf9 = ax9.pcolor(x_a1, y_a1, U4, cmap= 'seismic' , shading = 'auto')
fig9.colorbar(surf9)
ax9.set_ylim(-0.5, 0.5)
ax9.set_xlim(-0.2, 1)

fig91 = plt.figure()
ax91 = fig91.add_axes([0,0,1,1])
surf91 = ax91.pcolor(x_a1, y_a1, U5, cmap= 'seismic' , shading = 'auto')
fig91.colorbar(surf91)
ax91.set_ylim(-0.5, 0.5)
ax91.set_xlim(-0.2, 1)

fig92 = plt.figure()
ax92 = fig92.add_axes([0,0,1,1])
surf92 = ax92.pcolor(x_a1, y_a1, U6, cmap= 'seismic' , shading = 'auto')
fig92.colorbar(surf91)
ax92.set_ylim(-0.5, 0.5)
ax92.set_xlim(-0.2, 1)

fig93 = plt.figure()
ax93 = fig93.add_axes([0,0,1,1])
surf93 = ax93.pcolor(x_a1, y_a1, U7, cmap= 'seismic' , shading = 'auto')
fig93.colorbar(surf91)
ax93.set_ylim(-0.5, 0.5)
ax93.set_xlim(-0.2, 1)

fig10 = plt.figure()
ax10 = fig10.add_axes([0,0,1,1])
ax10.plot(Sigma / Sigma.sum(),'r-o')
ax10.set_xlabel('ith POD mode')
ax10.set_ylabel('$\Sigma_i / \Sigma_{sum}$')
ax10.set_title('Energy content per POD mode')
ax10.set_xlim(0,50)
ax10.grid(True)

fig11 = plt.figure()
ax11 = fig11.add_axes([0,0,1,1])
ax11.plot(E_tot / Sigma.sum(),'b-o')
ax11.set_xlabel('ith POD mode')
ax11.set_ylabel('Total energy of modes')
ax11.set_title('Energy content per POD mode')
ax11.set_xlim(0,50)
ax11.grid(True)