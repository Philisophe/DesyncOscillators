#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:57:56 2018

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

t = np.linspace(0,10000,100000)

# Integrate the finite number of Poincare oscillators using scipy.integrate.odeint and 
# return the container with integration points, which can be then plotted.

# Parameters of the function: 
# type of model we use (usually modified Poincare oscillator), 
# number of oscillators one want to integrate, initial conditions (states), 
# parameters of the equation given to odeint() function, timepoints.

# Returns list of arrays 'solutions', containing solutions for the oscillators as (x,y).
# Returns list of dictionaries 'analysisOfSolutions', containing some outputs of the analysis.
  
def odeP (model=model2, numOsc=1, state=[[1,1]], timepoints=np.linspace(0,100,1000)):
    solutions = [] # Storing variable for the solutions. solutions[0] corresponds to the solutions of the 0-th oscillator

    readouts = ['period', 'extrVal', 'extrT', 'zeroCrossInd', 'zeroCrossVal', 'zeroCrossT', 'mins', 'maxs', 'fold', 'SyncIndex', 'Variance']
    analysisOfSolutions = [] # list of dictionaries, containing the values of readouts for each solution
    for i in range(numOsc):
        params = [(0.1,1,(np.pi*2)/(24+ np.random.randn(1).tolist()[0]), 0)]
        solutions.append(odeint(model, state[i], timepoints, args = (params[0]))) #for each cycle turn, append the integrated values inside the solutions-list
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
                extrVal.append(np.mean(solutions[i][j:j+2,0]))
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
        """
        #  For some reason doesn't work correctly if twist is non-0
        if mins and maxs:  # If lists (mins and maxs) aren't empty, then
            for u in range(len(min(mins,maxs))):  # Take the smallest list (out of maxs and mins)
                fold.append(abs(maxs[u]/mins[u]))  # Then calculate the fraction max/min for a closest pair of maxs and mins, take the absolute value
"""                

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

###############################
###############################
    
state0 = []
x, anls = odeP(model=model2,numOsc=10,state=[[4,4]]*10,timepoints=t)
x1, anls1 = odeP(model=model2,numOsc=100,state=[[4,4]]*100,timepoints=t)

plt.figure(figsize=(12,5))
"""
for i in range(len(x)):
    plt.plot(t,x[i][:,0], 'o', label = 'x from x' + str(i))
"""
meanOsc=np.mean(x,axis=0)
plt.plot(t,meanOsc[:,0], 'o', label = 'mean values of 10 oscillators')

meanOsc1=np.mean(x1,axis=0)
plt.plot(t,meanOsc1[:,0], 'o', label = 'mean values of 100 oscillators')


plt.legend()
plt.show()

##################################
"""
Problems with this script:
    1) maxs, mins, and => fold don't work if twist is non-0
    2) You need to use the timepoint-variable in plt.plot() function, so no way of defining it inside the odeP() call
    3) You need to always use the correct number of initial states (state[]) and parameters (params=[()]).
Right now params are set to be the same for all different oscillators.
To fix it - put the exception inside, so if the len() of params, state[] and numOsc doesn't match - throw an exception
    4) 
"""
################################






