# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

def fib(w):
    a,b = 0,1
    result = []
    while(a<=w):
        result.append(a)
        a,b = b, a+b
    return result

"""
Let's make first of all the tonic neuron
"""
def tonicNeuron(length, stepsize):
    neuron = np.arange(0.0,length,stepsize) #Contains the time of the spike, it's position
    inst_freq = np.round(1/np.diff(neuron),2)
    plt.plot(inst_freq, '.')
    #print ("inst_freq is ", inst_freq)
    return inst_freq

def UniformNeuron(length,maxISI=1, refract=0):        #if maxISI=1 - ISI is between 0 and 1, uniformly distributed; if refract =0, then between 0 and 1, if refract=0.2, then from 0,2 to 1.2
    neuron = (npr.random(length)+refract)*maxISI #here it's different, it's not times of spikes, but rather an ISI - interspike interval
    inst_freq = np.round((1/neuron),4)
    plt.plot(inst_freq,'.')
    print (' inst_freq ',inst_freq)
    print ('neuron ', neuron)
    
def PoissonNeuron(length,maxISI=1, refract=0):
    neuron = (npr.random(length)+refract)*maxISI #here it's different, it's not times of spikes, but rather an ISI - interspike interval
    inst_freq = np.round((1/neuron),4)
    plt.plot(inst_freq,'.')
    print (' inst_freq ',inst_freq)
    print ('neuron ', neuron)