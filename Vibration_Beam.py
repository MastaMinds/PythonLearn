#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 09:40:32 2021

Python code for solving the FE model of a cantilever bear using state space 
modelling and implementation of a LQR controller and a state observer

@Authors: Ahmed Saeed Mohamed - Ahmed Osama Mahgoub
Qatar University
Ddepartment of Mechanical and Industrial Engineering
2021
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import Symbol
from scipy import linalg as SLA # Linear algebra library
import control # Control library

# Material Properties (Aluminum)
E = 69e9  # Elastic modulus (Pa)
rho = 2.7  # Material density (g/cm**3)
rho = rho/(1000/(100)**3)  # Material density (kg/m**3)
f = 1  # 1 N step load at the tip
tf = 1.5  # Final time
tspan = np.linspace(0, tf, 500)  # The time for the simulation


## Beam Properties
b = 0.2  # Section width (m)
t = 0.05  # Section thickness (m)
l = 2  # Beam length (m)
A = b*t  # Section area (m**2)
J = (b*t**3)/12  # Second Moment of Inertia (m**4) , Recatangular section
m_al = rho*A  # mass per unit length (kg/m)

## Finite Elements Discretization
m = 50  # Number of elements
n = 4  # Number of DOFs
NN = m*2 + 2
h = l/m  # Element length
xbeam = np.arange(0, l+h, h)  # Divisions of domain

# Element Matrices
# Each element has four DOFs, two DOFs per node (displacement and rotation)
# Equations 9.70 and 9.71 - File no. 1
Ke = ((E*J)/h**3)*np.array([[12, 6*h, -12, 6*h],
                            [6*h, 4*h**2, -6*h, 2*h**2],
                            [-12, -6*h, 12, -6*h],
                            [6*h, 2*h**2, -6*h, 4*h**2]])
Me = ((m_al*h)/420)*np.array([[156, 22*h, 54, -13*h],
                              [22*h, 4*h**2, 13*h, -3*h**2],
                              [54, 13*h, 156, -22*h],
                              [-13*h, -3*h**2, -22*h, 4*h**2]])
fe = np.transpose([[0, 0, f, 0]])

# Global matrices
# Matrix Assembly
M = np.zeros((NN, NN))
K = np.zeros((NN, NN))
F = np.zeros((NN, 1))

for i in range(m):
    # Le = Matrix for contribution of element in the global matrix
    # Based on equations 9.74 and 9.75 - File no. 1
    Le = np.zeros([4, NN])
    Le[0, 2*i-1] = 1 # Error is here
    Le[1, 2*i] = 1
    Le[2, 2*i+1] = 1
    Le[3, 2*i+2] = 1
    
    K = K + Le.transpose() @ Ke @ Le
    M = M + Le.transpose() @ Me @ Le

    if i == m:
        F = F + Le.transpose() @ fe
    else:
        F = F + Le.transpose() @ np.zeros([n, 1])


#%%
# Boundar condtions and modes
# Boundary conditions
# Eliminating first and 2nd rows and columns of both matrices
# Clamped end (vertical displacement and rotation = 0)
K = K[2: , 2: ]
M = M[2: , 2: ]
F = F[2: ]

# # Solving the generalized eigenvalue problem to find
# the approximate modes for the motion
# Omega = natural frequency of the mode
# U = Matrix of modes' eigenvectors
# Generalized eigenvalue problem
omegas, U = SLA.eigh(K, M) # Equations 10.96 and 10.98 - File no. 2
omega = np.sqrt(omegas)

# Modal matrices
M_m = np.transpose(U) @ M @ U
# M_m Modal mass
K_m = np.transpose(U) @ K @ U
# K_m Modal stiffness
F_m = np.transpose(U) @ F
# F_m Modal force

#%%
# Plotting mode shapes
fig1 = plt.figure()
ax1 = fig1.add_axes([0,0,1,1])
ax1.plot(xbeam,np.concatenate(([0], U[::2,0])))
ax1.plot(xbeam,np.concatenate(([0], U[::2,1])))
ax1.plot(xbeam,np.concatenate(([0], U[::2,2])))
ax1.plot(xbeam,np.concatenate(([0], U[::2,3])))
ax1.plot(xbeam,np.concatenate(([0], U[::2,4])))
ax1.set_xlabel('x (m)')
ax1.set_ylabel('deflection')
ax1.set_title('Cantilever beam vibration modes')
ax1.grid(True)
ax1.legend(['Mode 1','Mode 2','Mode 3','Mode 4','Mode 5'])

#%%
# # Damping matrix
# Damping model based on proportional damping
# C = alpha*M+Beta
# The coefficients selected in order to give damping ratio of 0.005 for the
# first and the 2nd modes
zeta_1 = 0.005
zeta_2 = 0.005
betac = (2*zeta_2*omega[1]-2*zeta_1*omega[0]) / (omega[1]**2-omega[0]**2)
alfac = 2*zeta_1*omega[0]-betac*omega[0]**2
C = alfac*M+betac*K # Equation 10.119 - File no. 2
C_m = alfac*M_m+betac*K_m

#%%
# Building a vector for zeta (modal damping coefficients)
nx = NN-2
zeta = np.zeros(nx)
omegad = np.zeros(nx)
for i in range(nx):
    zeta[i] = C_m[i, i]/(2*M_m[i, i]*omega[i])
    # Damped natural frequency
    omegad[i] = omega[i]*np.sqrt(1-zeta[i] ** 2) # Equation 10.17 - File no. 2

# State space representation (modal coordinates)
# Equations 14.23, 14.25, 14.34 - File no. 3
Lc = np.zeros((nx,1))
Lc[-2,0] = 1
A_sm = np.zeros((2*nx, 2*nx))
B_sm = np.zeros((2*nx,1)) # We need 2D arrays for State space system (column vector)
C_sm = np.zeros((1,2*nx)) # 2D array (row vector)
C_sm[0] = 1
for i in range(nx):
    A_sm[2*i : 2*(i+1), 2*i: 2*(i+1)] = [[0, 1],[- omega[i] ** 2, - 2*zeta[i]*omega[i]]]
    B_sm[2*i : 2*(i+1), 0] = np.array([0, U[:,i].transpose() @ Lc/M_m[i,i]]).transpose()
    C_sm[0, 2*i : 2*(i+1)] = [U[-2, i], 0]


# Finding the state space representation of the modal coordinates model
Cantsys_m = control.StateSpace(A_sm, B_sm, C_sm, 0)

# Reduced order model
# Only taking the first n_md modes
n_md = 1 # Number of modes to be included in the reduce model
A_r= A_sm[0:2*n_md, 0:2*n_md]
B_r= B_sm[0:2*n_md, 0]
C_r= C_sm[0, 0:2*n_md]

# Converting 1D arrays to 2D
B_r = B_r.reshape((2*n_md,1))
C_r = C_r.reshape((1,2*n_md))

Cantsys_r = control.StateSpace(A_r, B_r, C_r, 0)


#%% 
# LQR Controller Design
# Designing a controller to reduce the total energy
# Weights are equal to 1
Wz = 1 * np.eye(2*n_md)
Wu = 1
rho = 0.01
# The matrix H for obtaining the total energy
H = np.zeros((2*n_md,2*n_md))
for i in range(n_md):
    H[2*i-2:2*(i+1),2*i-2:2*(i+1)] = np.array([[omega[i]/np.sqrt(2), 0], [0, 1/np.sqrt(2)]])

# Finding the matrices Q and R
Q = np.transpose(H) @ Wz @ H # Should be matrix multiplication
R = rho*Wu
G, P, E  = control.lqr(A_r,B_r,Q,R)
Cantsys_con=control.StateSpace(A_r-B_r@G,B_r,C_r,0) # Equations 17.94 and 17.104

# Step response with and without LQR controller
T, Y = control.step_response(Cantsys_m,tspan)
Tc, Yc = control.step_response(Cantsys_con,tspan)

fig2 = plt.figure()
ax2 = fig2.add_axes([0,0,1,1])
ax2.plot(T,Y)
ax2.plot(Tc,Yc)
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Tip deflection (m)')
ax2.set_title('Step response of cantilever system')
ax2.grid(True)
ax2.legend(['Original system','With LQR controller'])


#%%  LQR Controller + Observer
# rhoo=0.01;
# Qo=eye(2*n_md);
# Ro=rhoo;
# LT=lqr(A_r',C_r',Qo,Ro);
# Lobs=LT';
#Observer=ss(A_r-B_r*G-Lobs*C_r,Lobs,-G,0);
# Cantsys_obs=feedback(Cantsys_m,Observer,+1);

# Simulation of controlled system
# xo=[0 1 zeros(1,2*(n_md-1))]';
# [y,tn,x]=initial(Cantsys_m,[xo;zeros(2*(nx-n_md),1)],tspan);
# [yc,tc,xc]=initial(Cantsys_con,xo,tspan);
# [yo,to,xo]=initial(Cantsys_obs,[xo;zeros(2*nx,1)],tspan);
# E=zeros(1,length(tn));
# u=zeros(1,length(tn));
# Ec=zeros(1,length(tc));
# uc=zeros(1,length(tc));
# Eo=zeros(1,length(to));
# uo=zeros(1,length(to));
# for i=1:length(tn)
#     E(i)=x(i,1:2*n_md)*H^2*x(i,1:2*n_md)';
#     u(i)=-G*x(i,1:2*n_md)';
# end
# for i=1:length(tc)
#     Ec(i)=xc(i,:)*H^2*xc(i,:)';
#     uc(i)=-G*xc(i,:)';
# end
# for i=1:length(to)
#     Eo(i)=xo(i,1:2*n_md)*H^2*xo(i,1:2*n_md)';
#     uo(i)=-G*xo(i,1:2*n_md)';
# end
# figure(2)
# subplot(2,1,1)
# plot(tn,E,tc,Ec,to,Eo)
# title('Total Energy vs time')
# xlabel('Time (seconds)')
# ylabel('Total Energy Nm')
# legend('W/o control',['+LQR \rho= ',num2str(rho)],['+Observer \rho= ',num2str(rhoo)])
# grid on
# subplot(2,1,2)
# plot(tc,uc,to,uo)
# title('Control force vs time')
# xlabel('Time (seconds)')
# ylabel('Control Force N')
# legend('LQR Controller','Controller+Observer')
# grid on

#%%




#%%


#%%
