# function that returns dy/dt

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


def oscillator(x, y, t, i, alpha, A, omega, twist, K): 
    
    x1 = x[i]
    y1 = y[i]
    dx1dt = x1*alpha*(A-np.sqrt(x1**2 + y1**2)) - y1*(omega + twist*(A - np.sqrt(x1**2 + y1**2))) + K*(np.mean(x))
    dy1dt = y1*alpha*(A-np.sqrt(x1**2 + y1**2)) + x1*(omega + twist*(A - np.sqrt(x1**2 + y1**2)))

    return dx1dt, dy1dt


def oscillator_system(state_vector, t, alpha, A, omega, twist, K):

    # It receives initial conditions as a list in form [x1,y1,x2,y2,x3,y3,x4,y4,...], where x1,y1 are initial conditions for the 1st oscillator
    # And converts it into array with 2 columns
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
        dzdt[i, ] = oscillator(x, y, t, i, alpha, A, omega, twist, K)

    return dzdt.reshape(-1).tolist()



t = np.linspace(0, 500, 1000)
params = (0.1,1,(np.pi*2)/24, 0.2, 0.05) #Turn twist up just a bit, and you'll see a very different pic
state0 = [1,1,2,2,4,4]

x1 = odeint(oscillator_system, state0, t, args = (params))


n = int(len(state0) / 2)

plt.figure(figsize=(20,8))
for i in range(n):
    plt.plot(x1[:, 2*i], label = 'x coord of {}st osc'.format(i))
plt.ylim(-2,5)
plt.legend()
plt.show()