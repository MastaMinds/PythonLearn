# -*- coding: utf-8 -*-
"""
Ahmhmd functions library
"""

import numpy as np
from scipy import interpolate

def read_OpenFOAM_Udata(filename, N_probes, N_header):
    
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    # Only recording values
    Lines = Lines[N_probes + N_header:]
    n_lines = len(Lines)
    
    translation_table = dict.fromkeys(map(ord, '(){}!@#$\n'), None)
    Lines_new = n_lines * [0]
    
    k=0
    for line in Lines:
        Lines_new[k] = line.translate(translation_table).split()
        Lines_new[k] = [float(i) for i in Lines_new[k]]
        k = k + 1
        
    Lines_new = np.array(Lines_new)
    t = Lines_new[:,0]
    data = Lines_new[:,1:]
    
    return t, data

def getTurbulenceQauntities(Vel):
    U = Vel[:,0]
    V = Vel[:,1]
    W = Vel[:,2]
    Vmag = np.sqrt(U**2 + V**2 + W**2)
    u = U - U.mean()
    v = V - V.mean()
    w = W - W.mean()
    
    uu = (u**2).mean()
    vv = (v**2).mean()
    ww = (w**2).mean()
    uv = (u*v).mean()
    uw = (u*w).mean()
    vw = (v*w).mean()
    
    R = np.array([[uu, uv, uw],
                  [uv, vv, vw],
                  [uw, vw, ww]])
    return u, v, w, R

def get2DGridData(x, y, C,  x_a1, y_a1):
    # x_a and y_a = [start end step]
    y_a, x_a = np.mgrid[y_a1[0]:y_a1[1]:y_a1[2], x_a1[0]:x_a1[1]:x_a1[2]]
    C_2D = interpolate.griddata((x,y), C, (x_a, y_a), method='linear')
    
    return x_a, y_a, C_2D
