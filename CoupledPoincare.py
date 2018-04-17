#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:59:17 2018

@author: kalashnikov
"""

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# function that returns dy/dt
def model2 (vector,t,alpha, A, omega, twist, K):
#    for i in range(numOsc):
#        x
    x1 = vector[0]
    y1 = vector[1]
    x2 = vector[2]
    y2 = vector[3]
    
    dx1dt = x1*alpha*(A-np.sqrt(x1**2 + y1**2)) - y1*(omega + twist*(A - np.sqrt(x1**2 + y1**2))) + K*((x1+x2)/2)
    dy1dt = y1*alpha*(A-np.sqrt(x1**2 + y1**2)) + x1*(omega + twist*(A - np.sqrt(x1**2 + y1**2)))
    
    dx2dt = x2*alpha*(A-np.sqrt(x2**2 + y2**2)) - y2*(omega + twist*(A - np.sqrt(x2**2 + y2**2))) + K*((x1+x2)/2)
    dy2dt = y2*alpha*(A-np.sqrt(x2**2 + y2**2)) + x2*(omega + twist*(A - np.sqrt(x2**2 + y2**2)))
    
    dzdt = [dx1dt, dy1dt,dx2dt,dy2dt]
    return dzdt

t = np.linspace(0,100,1000)

params = (0.1,1,(np.pi*2)/24, 0.3, 0.5) #Turn twist up just a bit, and you'll see a very different pic
state0=[4,4,1,1]

x1 = odeint(model2, state0, t, args = (params))
plt.figure(figsize=(20,8))
plt.plot(t, x1[:,0], label = 'x coord of 1st osc')
plt.plot(t, x1[:,2], label = 'x coord of 2nd osc')


plt.legend()
plt.show()

"""
So, I don't see the easy way to write down 20 equations (10 sets of them) without explicitly writing them.

If I choose the explicit way of writing the equations, I need to somehow find a way to assign for each set of equations a set of parameters.
The only way I can do it with an explicit notation, is by giving each parameter it's own name (omega1, omega2, omega3 etc.), 
so eventually the execution will look like this:
    params = (0.1,1,omega1,omega2, omega3, ..., twist1, twist2, twist3, ..., K1, K2, K3 ...)
    x = odeint(model2,state0,t,args=(params))
"""
