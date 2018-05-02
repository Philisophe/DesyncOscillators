#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 12:25:52 2018

@author: kalashnikov
"""

from scipy.integrate import odeint
from scipy.signal import hilbert
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt




def oscillator(x, y, t, i, alpha, A, omega, twist, K, E): 
    """ Complementary function for solver (oscillator_system) needed to solve multiple coupled (having terms that refer to each other) equations simultaneously.
In case you have a low number of equations (up to 4-6) it might be OK to just explicitely set the equations and parameters in the solver itself. 
However, if you want tp compute more equations, as well as set the values for every parameter separately and easily change the number of equations you have, this code might help you.

Each coordinate (x,y) or parameter (alpha,A,omega,twist,K,E) is a vector x[], where x[i] is a value for i-th oscillator.
It takes vectors as input and returns vectors as output. 

The system (coupled Poincare oscillators with twist and noise) is:
dx1dt = x1*alpha1*(A1-np.sqrt(x1**2 + y1**2)) - y1*(omega1 + twist1*(A1 - np.sqrt(x1**2 + y1**2))) + K1*(np.mean(x)) + E1
dy1dt = y1*alpha1*(A1-np.sqrt(x1**2 + y1**2)) + x1*(omega1 + twist1*(A1 - np.sqrt(x1**2 + y1**2)))

"""

    # It takes x and y (as well as parameters) as vectors, then throws out dx/dt and dy/dt for every respective value
    x1 = x[i]
    y1 = y[i]
    
    alpha1 = alpha[i]
    A1 = A[i]
    omega1 = omega[i]
    twist1=twist[i]
    K1 = K[i]
    E1 = E[i]
    
    dx1dt = x1*alpha1*(A1-np.sqrt(x1**2 + y1**2)) - y1*(omega1 + twist1*(A1 - np.sqrt(x1**2 + y1**2))) + K1*(np.mean(x)) + E1
    dy1dt = y1*alpha1*(A1-np.sqrt(x1**2 + y1**2)) + x1*(omega1 + twist1*(A1 - np.sqrt(x1**2 + y1**2)))

    return dx1dt, dy1dt









# Solver of ODEs
def oscillator_system(state_vector, t, alpha, A, omega, twist, K, E):

    """ This function describes the 1st parameter for the odeint() function. It uses oscillator() and can be used by ode_rand().
    
    It takes initial conditions (state_vector) as a list in form [x1,y1,x2,y2,x3,y3,x4,y4,...], 
    where x1,y1 are initial conditions for the 1st oscillator and returns vector of results.
    """
    state_mat = np.array(state_vector).reshape(-1, 2)
    
    # Then we take only the 1st and only the 2nd column and put them into separate variables
    x = state_mat[:, 0]
    y = state_mat[:, 1]
    
    #n being the number of rows in state_map; in other words - number of sets of initial conditions, number of oscillators
    n = state_mat.shape[0]
    # Let's produce an array of the same shape as state_map
    dzdt = np.zeros((n, 2))
    
    #For every i-th row of state_map let's put there dx1/dt, dy1/dt 
    for i in range(n):
        dzdt[i, ] = oscillator(x, y, t, i, alpha, A, omega, twist, K, E)

    return dzdt.reshape(-1).tolist()



#params = ([0.1]*n,[1]*n,[(np.pi*2)/(24 + 2*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)


##################################################


def ode_rand2(number_of_oscillators, iterations, timepoints, state0, params, randMulti):   
    """The function models the behaviour of system of coupled Poincare oscillators with noise. 
    To do that, it executes odeint() function with oscillator_system as a first parameter multiple times in a row, changing each time noisy variable E to a random value drawn from standart normal distribution (SND).
    The dispersion (sigma) of SND for E is set by randMulti parameter.
    
    timepoints - expects a tuple (timestart, timeend, number_of_timepoints)
    Please, always set timepoints[0] to 0.
    
    The function returns solutions in the form of np.array. The length of the array is len(timepoints)*iterations.
    
    Example of execution: 
    2 oscillator system executed 160 consequetive times with 10 datapoints each, starting from [2,2] and [3,3] with params as all parameters except for noise, which is set explicitely by E.
    
    x4=ode_rand2(2,160,(0,0.5,10),[2,2,3,3],params,0.1)
    plt.plot(x4[:,0], label = 'x-coordinate of the 1st oscillator')
    plt.legend()
    
    """
    # Unpacking the 'timepoints' parameter
    timestart = timepoints[0]
    timeend = timepoints[1]
    number_of_timepoints = timepoints[2]
    
    n = number_of_oscillators # Shortcut
    solutions = np.zeros((number_of_timepoints*iterations-iterations+1,n*2)) # Creates array of zeros of an appropriate size to store iterative executions of odeint() function
    
    t = np.linspace(timestart, timeend, number_of_timepoints) # First timepoint-variable
    time = [] # Variable for storing the timepoints from all t
    
    start=0
    end=len(t) # Initial start and end for the overwriting of solutions
    
    
    for i in range(iterations):
        
        E = randMulti*np.random.randn(n) # Creates vector of random numbers from SND
        
        s = odeint(oscillator_system, state0, t, args = ((params[0], params[1], params[2], params[3], params[4], E))) # The parameters: alpha, amplitude, omega, twist, coupling
        time.append(list(t))
        solutions[start:end] = s
        
        state0 = s[-1].tolist()
        
        start = end-1
        end += len(t)-1
        
        timestart, timeend = timeend, (timeend+timeend-timestart)
        t = np.linspace(timestart, timeend, number_of_timepoints) # Changing t variable to the new timestart and timeend
        
        
        
    # "time" is a list of lists, so it should be flattened
    # because of the overlap (last element of the previoud iteration of odeint() being the first element of the new iteration of odeint())
    # The consecutive duplicates need to be removed
    return remdup(flat_list(time)), solutions










def flat_list(l):
    """Unpacks list of lists -> flattened list"""
    return [item for sublist in l for item in sublist]


def remdup(x):
    """Remove consecutive duplicates from the list"""
    i=0
    while i < len(x)-1:
        if x[i] == x[i+1]:
            del x[i]
        else:
            i = i+1
    return x


def npremdup(x):
    """Remove consecutive duplicates from the np.array"""
    i=0
    while i < len(x)-1:
        if x[i] == x[i+1]:
            x = np.delete (x,i)
        else:
            i = i+1
    return x



def extr(x):
    """Finds the extrema of the function. Returns [timepoints of extrema, values of extrema] list"""
    diff = np.diff(np.sign(np.diff(x)))
    extrT=[]
    extrVal=[]
    for i in range(len(diff)):
        if diff[i]!=0:
            extrVal.append(np.mean(x[i:i+2]))
            extrT.append(np.mean(t[i:i+2]))
    return [extrT,extrVal]


def maxs(list_extr):
    """Returns every odd element. Designed to be used in combination with extr(), e.g. maxs(extr(solutions))"""
    maxsV=[]
    maxsT=[]
    for i in range(int(len(list_extr[0])/2)):
        maxsV.append(list_extr[1][i*2 + 1])
        maxsT.append(list_extr[0][i*2 + 1])
    return [maxsT,maxsV]


def me(x):
    return maxs(extr(x))
def me2(x):
    return me(np.mean(x, axis=0))


def run_mean(x, N, N2=0):
    """Running average
    x - data, 
    N - window size, 
    N2 - number of runs (if N2=0 - 1 run, if N2=1 - 2 runs etc.)"""
    if N2==0:
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    else:
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return run_mean(((cumsum[N:] - cumsum[:-N]) / float(N)),N,N2-1)


def cart2pol(x, y):
    """Transforms cartesian coordinates to polar coordinates; 
    returns vector [theta,rho], where theta is angle in degrees"""
    theta = np.rad2deg(np.arctan2(y, x)) # ix1pol = [cart2pol(x1[:,i*2],x1[:,i*2 +1]) for i in range(n)]n degrees
    rho = np.hypot(x, y)
    return [theta, rho]


# Separates x- and y- coordinates
# int(np.shape(solution)[1]/2)) is the same as n
def sep(solution):
    """Separates x- and y- coordinate data, received from odeint() function"""
    solution_x = [solution[:,i*2] for i in range(int(np.shape(solution)[1]/2))]
    solution_y = [solution[:,i*2+1] for i in range(int(np.shape(solution)[1]/2))]
    return [solution_x,solution_y]



def sol2pol(solution):
    """Translate the result of odeint() function into the polar coordinates and returns only the positive values"""
    solution_pol = [cart2pol(solution[:,i*2],solution[:,i*2+1]) for i in range(int(np.shape(solution)[1]/2))]
    for j in range(int(np.shape(solution)[1]/2)):
        solution_pol[j][0] = abs(solution_pol[j][0]) # Only positive values for phases
    return solution_pol



# Phase (theta) variance
# Returns phase variance and the upper envelope of it
# Looks ugly, but that's OK; relies on 't'
def phvar(solution):
    sol_pol = sol2pol(solution)
    thetas = [sol_pol[i][0] for i in range(int(np.shape(solution)[1]/2))]
    var = np.var(thetas, axis=0)
    extrVal=[]
    extrT=[]
    diff = np.diff(np.sign(np.diff(var)))
    for j in range(len(diff)):
        if diff[j]!=0:
            extrVal.append( (np.mean(var[j:j+2])) )
            extrT.append( (np.mean(t[j:j+2])) )
    # Taking only the positive part of the envelope
    extrVal2 = [extrVal[i*2] for i in range(int(len(extrVal)/2))]
    extrT2 = [extrT[i*2] for i in range(int(len(extrVal)/2))]
    return var,[extrT2,extrVal2]



def env(x):
    """Calculates envelope of the time-series using Hilbert transformation"""
    return np.abs(hilbert(x))

# Some standart functions to fit to
def lin(x, a, b):
    return (a*x + b)

def quad(x, a, b, c):
    return (a*(x**2) + b*x + c)

def cub(x,a,b,c,d):
    return (a*(x**3) + b*(x**2) + c*x + d)

def expon(x, a, b, c):
    return a * np.exp(-b * x) + c





"""
"""




n=2 
params = ([0.1]*n,[1]*n,[(np.pi*2)/24]*n,[0.0]*n,[0.0]*n) 
# alpha (amplitude relaxation rate), A (amplitude), omega, twist, K (coupling), E (white noise, if any)




