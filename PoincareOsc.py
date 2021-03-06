#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 17:37:23 2018

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

# time points
stepsize = 1
hours = 30
start = 0        
t = np.linspace(start,hours,hours*stepsize) # generates the timepoints from start, hours-long with a stepsize
#notice, that np.linspace(start, hours, stepsize) doesn't really provide you with a stepsize you want it to be


# initial conditions
state0 = [1,1]
state01 = [0,1]
state02 = [4,4]
state03 = [1.5,1]

# sets of parameters
params = (0.1,1,(np.pi*2)/24, 0.5) #relaxation rate (alpha/lambda); amplitude (A); angle speed (omega); twist
params1 = (1,1,(np.pi*2)/24, 0)
params2 = (0.5,1,(np.pi*2)/24, 0.5)
params3 = (1,1,(np.pi*2)/24, 0.5)
params4 = (0.1,1,(np.pi*2)/24, 0.5)

# solves ODEs
#here are ODEs with different parameters but same initial conditions
x0 = odeint(model2, state02, t, args=(params))
x1 = odeint(model2, state02, t, args=(params1))
x2 = odeint(model2, state02, t, args=(params2))
x3 = odeint(model2, state02, t, args=(params3))
x4 = odeint(model2, state02, t, args=(params4))



# Integrate the finite number of Poincare oscillators using scipy.integrate.odeint and 
# return the container with integration points, which can be then plotted.

# Parameters of the function: 
# type of model we use (usually modified Poincare oscillator), 
# number of oscillators one want to integrate, initial conditions (states), 
# parameters of the equation given to odeint() function, timepoints.

# Returns list of arrays 'solutions', containing solutions for the oscillators as (x,y).
# Returns list of dictionaries 'analysisOfSolutions', containing some outputs of the analysis.
  
def odeP (model=model2, numOsc=1, state=[[1,1]], timepoints=np.linspace(0,100,100), params=[(0.1,1,(np.pi*2)/24,0)]):
    solutions = [] # Storing variable for the solutions. solutions[0] corresponds to the solutions of the 0-th oscillator

    readouts = ['period', 'extrVal', 'extrT', 'zeroCrossInd', 'zeroCrossVal', 'zeroCrossT', 'mins', 'maxs', 'fold', 'SyncIndex', 'Variance']
    analysisOfSolutions = [] # list of dictionaries, containing the values of readouts for each solution
    for i in range(numOsc):
        solutions.append(odeint(model, state[0], timepoints, args = (params[0]))) #for each cycle turn, append the integrated values inside the solutions-list
        analysisOfSolutions.append(dict.fromkeys(readouts)) #
       
        #the indices of x-values just before the crossings
        zeroCrossInd = np.where(np.diff(np.sign(solutions[i][:,0])))[0]
        
        # the values themselves, should be really close to 0
        zeroCrossVal = (solutions[i][zeroCrossInd,0]+solutions[i][zeroCrossInd+1,0])/2
        
        #the approximate times of actual crossings
        zeroCrossT = (timepoints[zeroCrossInd]+timepoints[zeroCrossInd+1])/2 # takes the time in between two x-values of opposing signs
        period = np.diff(zeroCrossT)
        
        # Storing variables for local extrema
        extrVal = [] # values
        extrT = [] # timepoints for respective values

        # Looking for local maxima, minima
        # When result of np.diff() changes the sign - it's when the max of min occured 
        diff = np.diff(np.sign(np.diff(solutions[i][:,0])))
        for j in range(len(diff)):
            if diff[j]!=0:
                extrVal.append(np.mean(solutions[j:j+2,0]))
                extrT.append(np.mean(timepoints[j:j+2]))
        
        # different lists for minima and maxima  
        # Can only differentiate them if they have different sign
        mins=[]
        maxs=[]
        for k in extrVal:
            if k<0:
                mins.append(k)
            else:
                maxs.append(k)
        
        fold = []
        if mins and maxs:  # If lists (mins and maxs) aren't empty, then
            for u in range(len(min(mins,maxs))):  # Take the smallest list (out of maxs and mins)
                fold.append(abs(maxs[u]/mins[u]))  # Then calculate the fraction max/min for a closest pair of maxs and mins, take the absolute value
                
        analysisOfSolutions[i]['zeroCrossInd'] = zeroCrossInd
        analysisOfSolutions[i]['zeroCrossVal'] = zeroCrossVal
        analysisOfSolutions[i]['zeroCrossT'] = zeroCrossT
        analysisOfSolutions[i]['period'] = period        
        analysisOfSolutions[i]['extrVal'] = extrVal
        analysisOfSolutions[i]['extrT'] = extrT
        analysisOfSolutions[i]['mins'] = mins
        analysisOfSolutions[i]['maxs'] = maxs
        analysisOfSolutions[i]['fold'] = fold

    
    return solutions, analysisOfSolutions


# Here we extract the 0-crossings (only changes from + to -, without from - to +), mins and maxes

# Should I use y or x when finding the 0-crossings, mins and maxs? 
# Does it even matter? I wouldn't think it does
#the indices of x-values just before the crossings
zeroCrossInd = np.where(np.diff(np.sign(x0[:,0])))[0]
# the values themselves, should be really close to 0
zeroCrossVal = (x0[zeroCrossInd,0]+x0[zeroCrossInd+1,0])/2
#the approximate times of actual crossings
zeroCrossT = (t[zeroCrossInd]+t[zeroCrossInd+1])/2 # takes the time in between two x-values of opposing signs
period = np.diff(zeroCrossT)





# Local extrema
extrVal = [] # values
extrT = [] # timepoints for respective values

# Looking for local maxima, minima
# Maybe np.diff(np.sign(np.diff(x0[:,0])))? When result of np.diff() changes the sign - it's when the max of min occured 
diff = np.diff(np.sign(np.diff(x0[:,0])))
for i in range(len(diff)):
    if diff[i]!=0:
        extrVal.append(np.mean(x0[i:i+2,0]))
        extrT.append(np.mean(t[i:i+2]))





# different lists for minima and maxima        
mins=[]
maxs=[]
for i in extrVal:
    if i<0:
        mins.append(i)
    else:
        maxs.append(i)

fold = maxs[-1]/mins[-1]



# _____________________________________________
# _____________________________________________
# PLOTTING
# time-series
plt.figure(figsize=(12,5))
plt.plot(t, x0[:,0], 'o', label = 'x from x0')
#plt.plot(t, x4[:,0], label = 'x from x4')
plt.plot(zeroCrossT, np.zeros(len(zeroCrossT)), 'r+', label = 'zero crossings of x-coordinate of x0') 
plt.plot(extrT,extrVal,'-v')
plt.xlim(start-1, hours-start+1) #shows the x from a little bit before the start of timepoints, to a little bit after the end
plt.ylim(-2.5,4.5)

plt.legend()
plt.show()







"""
So the next step is to identify the functions that need to be "functionized", in order o separate the methods from the data itself
"""







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






"""
# Another implementation of zeroCross
zeroCross=[]
for i in range(len(x0[:,0]-1)):
    if (x0[i,0]>0 and x0[i+1,0]<=0):
        zeroCross.append((t[i]+t[i+1])/2) # we append to zeroCross the approx. time of 0-crossing

#here are same ODEs with different initial conditions
x0 = odeint(model2, state0, t, args=(params))
x1 = odeint(model2, state01, t, args=(params))
x2 = odeint(model2, state02, t, args=(params))
x3 = odeint(model2, state03, t, args=(params))


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










