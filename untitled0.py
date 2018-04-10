#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:52:35 2018

@author: kalashnikov
"""
import matplotlib.pyplot as plt
import numpy as np


def odeP (model=model2, numOsc=1, state=[[1,1]], timepoints=np.linspace(0,100,100), params=[(0.1,1,(np.pi*2)/24,0)]):
    solutions = [] # Storing variable for the solutions. solutions[0] corresponds to the solutions of the 0-th oscillator

    readouts = ['period', 'extrVal', 'extrT', 'zeroCrossInd', 'zeroCrossVal', 'zeroCrossT', 'mins', 'maxs', 'fold', 'SyncIndex', 'Variance']
    analysisOfSolutions = [] # list of dictionaries, containing the values of readouts for each solution
    for i in range(numOsc):
        solutions.append(odeint(model, state[0], timepoints, args = (params[0])))
         #
    return [solutions, analysisOfSolutions]
