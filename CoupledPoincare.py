#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:59:17 2018

@author: kalashnikov
"""

from scipy.integrate import odeint
import numpy as np
#import matplotlib.pyplot as plt

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

params = (0.1,1,(np.pi*2)/24, 0.5, 0)
state0=[4,4]*2
x1 = odeint(model2, state0, t, args = (params))\

