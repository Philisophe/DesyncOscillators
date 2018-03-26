#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 17:37:23 2018

@author: kalashnikov
"""
from scipy.integrate import odeint
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
"""
def model(y,t,k):
    dydt = -k * y
    return dydt

# initial condition
y0 = 5

# time points
t = np.linspace(0,20)

# solve ODEs
k = 0.1
y1 = odeint(model,y0,t,args=(k,))
k = 0.2
y2 = odeint(model,y0,t,args=(k,))
k = 0.5
y3 = odeint(model,y0,t,args=(k,))

# plot results
plt.plot(t,y1,'r-',linewidth=2,label='k=0.1')
plt.plot(t,y2,'b--',linewidth=2,label='k=0.2')
plt.plot(t,y3,'g:',linewidth=2,label='k=0.5')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()
"""

"""
# function that returns dy/dt
def model2 (x,y,t, A,alpha,omega,twist):
    dxdt = x*alpha*(A-np.sqrt(x**2 + y**2))-y*(omega + twist*(A-np.sqrt(x**2 + y**2)))
    dydt = y*alpha*(A-r)+x*(omega + twist*(A-r))
    dzdt=[dxdt, dydt]
    return dzdt

# time points        
t = np.linspace(0,20)

# initial conditions
x0=1
y0=1

A, r, alpha, omega, twist = 1, 1, 0.5, (2*np.pi)/24, 0

#x1 = odeint(model2, x0, y0, t, args=(A, alpha,omega,twist))
#y1 = odeint(model2, x0, y0, t, args=(A,r, alpha,omega,twist,))
x1 = odeint(model2, x0, y0, t)


plt.plot(y1,x1,'r-', linewidth=2)
plt.xlabel('time')
plt.ylabel('')
plt.show()
"""
# function that returns dy/dt

def model2 (vector,t,alpha, A, omega, twist):
    x = vector[0]
    y = vector[1]
    dxdt = x*alpha*(A-np.sqrt(x**2 + y**2)) - y*(omega + twist*(A - np.sqrt(x**2 + y**2)))
    dydt = y*alpha*(A-np.sqrt(x**2 + y**2)) + x*(omega + twist*(A - np.sqrt(x**2 + y**2)))
    dzdt = [dxdt, dydt]
    return dzdt

# time points
stepsize = 1
hours = 24
start = 0        
t = np.linspace(start,hours,hours*stepsize) # generates the timepoints from start, hours-long with a stepsize
#notice, that np.linspace(start, hours, stepsize) doesn't really provide you with a stepsize you want it to be


# initial conditions
state0 = [1,1]
state01 = [0,1]
state02 = [2,2]
state03 = [1.5,1]

# sets of parameters
params = (0.5,1,(np.pi*2)/24, 0) #relaxation rate (alpha/lambda); amplitude (A); angle speed (omega); twist
params1 = (1,1,(np.pi*2)/24, 0)
params2 = (0.5,1,(np.pi*2)/24, 0.5)
params3 = (1,1,(np.pi*2)/24, 0.5)
params4 = (0.1,1,(np.pi*2)/24, 0.5)

# solves ODEs

#here are same ODEs with different initial conditions
"""x0 = odeint(model2, state0, t, args=(params))
x1 = odeint(model2, state01, t, args=(params))
x2 = odeint(model2, state02, t, args=(params))
x3 = odeint(model2, state03, t, args=(params))
"""
#here are ODEs with different parameters but same initial conditions
x0 = odeint(model2, state02, t, args=(params))
x1 = odeint(model2, state02, t, args=(params1))
x2 = odeint(model2, state02, t, args=(params2))
x3 = odeint(model2, state02, t, args=(params3))
x4 = odeint(model2, state02, t, args=(params4))



# time-series
plt.figure(figsize=(12,5))
plt.plot(t, x0[:,0], label = 'x from x0')
plt.plot(t, x4[:,0], label = 'x from x4')
plt.xlim(start-1, hours-start+1) #shows the x from a little bit before the start of timepoints, to a little bit after the end
plt.ylim(-2.5,4.5)


"""
# Phase-plots
plt.figure(figsize=(10,10))
plt.plot(x0[:,0],x0[:,1],'r-', linewidth=2, label = 'relax.rate = 0.5, twist=0')
plt.plot(x1[:,0],x1[:,1],'g-', linewidth=2, label = 'relax.rate = 1, twist=0')
plt.plot(x2[:,0],x2[:,1],'b-', linewidth=2, label = 'relax.rate = 0.5, twist=0.5')
plt.plot(x3[:,0],x3[:,1],'k-', linewidth=2, label = 'relax.rate = 1, twist=0.5')
#plt.plot(x4[:,0],x4[:,1],'m-', linewidth=2, label = 'relax.rate = 0.1, twist=0.5')

plt.xlim(-1.5,2.5)
plt.ylim(-1.5,2.5)
plt.xlabel('x(t)')
plt.ylabel('y(t)')
"""

plt.legend()
plt.show()


# STRANGE THINGS:
# ---------------
# 1) 
# Why even stepsize=1 is enough to produce relatively smooth oscillation? 
# I mean, with 10 you don't see any sudden changes, so with 100 it should be alright, 
# but I would expect to see a lot of problems with stepsize=1

# 2) 
# Why should I even think about power spectra? 
# And which spectra precisely am I interested in and why? E.g., power or energy, periodogram, if so - which window type?
# What's the point of plotting the spectra of the oscillations with known period and known frequency? 
# I mean, with coupled oscillators, it could be complex enough, but otherwise?

# 3) 
# Is there any sence to plot graphs with different amplitude? 
# It just changes how big the limit cycle is (r=1,2,3 ...), nothing else.

#______________________________________________

# TO-DO for tomorrow, 27th March
# 1) Extract the 0-crossings, maxs and mins as 3 different lists to compute then magnitude.
# 2) Find some nice way to make a lot of graphs without changing everything in all of them
#  2.a) Labels should automatically reflect the changes in the variables.
#  2.b) Find how you can make a plot bigger, jusr bigger in all dimensions, witout manually assigning plt.figure(figsize=(x,y))
#  2.c) plt.ylim and plt.xlim should automatically be a little bit larger than the max and min values.
#  2.d) Phase portraits should be plotted simultaneously with time-series and magnitude/period/phase, 
#       maybe on the same graphs, maybe on the different, that needs to be thought through.
#  2.e) Get rid of "Reloaded modules".


















