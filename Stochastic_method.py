#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:00:43 2018

@author: kalashnikov
"""

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# function that returns dy/dt
def model2 (vector,t,alpha, A, omega, twist):
    x = vector[0]
    y = vector[1]
    dxdt = x*alpha*(A-np.sqrt(x**2 + y**2)) - y*(omega + twist*(A - np.sqrt(x**2 + y**2)))
    dydt = y*alpha*(A-np.sqrt(x**2 + y**2)) + x*(omega + twist*(A - np.sqrt(x**2 + y**2)))
    dzdt = [dxdt, dydt]
    return dzdt

t2 = np.linspace(0,0.5,10)
params = (0.1,1,(np.pi*2)/24, 0.5)
state0=[4,4]
odeint(model2, state0, t2, args = (params))




# For beginning I only have 1 noisy coordinate (x), not both
# randMulti tells you which dispersion should random variable E have
def ode_rand(iterations=1, timepoints=np.linspace(0,0.5,10), state0=[4,4], params=(0.1,1,(np.pi*2)/24, 0.5), randMulti=1):
    solutions = np.empty(shape=[0,2]) 
    # Storing variable for solutions. Time-inefficient solution 
    # (better - create 1 time an array via np.zeros(), and then set the values to smth.else)

# The loop works like this: draw number E from normal standart distribution, solve the model with parameters and this E, 
# change the initial state to the last solution of the model, repeat.
    for i in range(iterations): 
        E = randMulti*float(np.random.randn(1)) # Noisy variable
        def model2 (vector,timepoints,alpha, A, omega, twist):
            x = vector[0]
            y = vector[1]
            dxdt = x*alpha*(A-np.sqrt(x**2 + y**2)) - y*(omega + twist*(A - np.sqrt(x**2 + y**2))) + E
            dydt = y*alpha*(A-np.sqrt(x**2 + y**2)) + x*(omega + twist*(A - np.sqrt(x**2 + y**2)))
            dzdt = [dxdt, dydt]
            return dzdt
        solutions = np.append(solutions, odeint(model2,state0,timepoints,args=(params)), axis=0) # Works here like .append() for lists as storing variables
        state0 = solutions[-1].tolist()

    return solutions


# Solving
sol = ode_rand(500,t2,[4,4],params, 0.01)
sol2 = ode_rand(500,t2,[4,4],params, 0.05)
sol3 = ode_rand(500,t2,[4,4],params, 0.1)
sol4 = ode_rand(500,t2,[4,4],params, 0.2)
sol5 = ode_rand(500,t2,[4,4],params, 0.5)

# Plotting
plt.figure(figsize=(19,8))

#plt.plot(sol[:,0], label='E from 0.01*(st.norm.distr.)')
plt.plot(sol2[:,0], label='E from 0.05*(st.norm.distr.)')
plt.plot(sol3[:,0], label='E from 0.1*(st.norm.distr.)')
#plt.plot(sol4[:,0], label='E from 0.2*(st.norm.distr.)')
plt.plot(sol5[:,0], label='E from 0.5*(st.norm.distr.)')


plt.legend()
plt.show()