# -*- coding: utf-8 -*-
"""
Python code for obtaining the POD modes for a 2D laminar flow
past a Cylinder

Made by Ahmhmd UNSW - 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d
import pandas as pd

Main_path = 'C:/Users/ahmhm/POD_Cylinder/Cylinder_2D_Laminar/'

dt = 0.1
T_final = 100
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
    filenames[i] = Main_path + 'Cylinder_2D_Foam_' + str(i+1) + '.csv'
    data_all[i] = pd.read_csv(filenames[i])
    
    # Append data matrix using current snapshot data
    X[:,i] = data_all[i]['U:0']
    # X[:,i] = data_all[i]['vorticity:2']
    print('Finished data file number'+str(i+1))
    

x = test_table['Points:0']
y = test_table['Points:1']
z = test_table['vorticity:2']