# This code integrates over a system of ODEs of coupled oscillators with parameters.
# The single oscillator is implemented as a modified Poincare oscillator (with twist - parameter, linking period and amplitude).


from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


# Complementary function for solver (oscillator_system) needed to solve multiple coupled (having terms that refer to each other) equations simultaneously.
# In case you have a low number of equations (up to 4-6) it might be OK to just explicitely 
# set the equations and parameters in the solver itself. However, if you want tp compute more equations, as well as set the values for every parameter separately and easily change the number of equations you have,
# This code might help you.

def oscillator(x, y, t, i, alpha, A, omega, twist, K): 
    # It takes x and y (as well as parameters) as vectors, then throws out dx/dt and dy/dt for every respective value
    x1 = x[i]
    y1 = y[i]
    
    alpha1 = alpha[i]
    A1 = A[i]
    omega1 = omega[i]
    twist1=twist[i]
    K1 = K[i]
    
    dx1dt = x1*alpha1*(A1-np.sqrt(x1**2 + y1**2)) - y1*(omega1 + twist1*(A1 - np.sqrt(x1**2 + y1**2))) + K1*(np.mean(x))
    dy1dt = y1*alpha1*(A1-np.sqrt(x1**2 + y1**2)) + x1*(omega1 + twist1*(A1 - np.sqrt(x1**2 + y1**2)))

    return dx1dt, dy1dt


# Solver of ODEs
def oscillator_system(state_vector, t, alpha, A, omega, twist, K):

    # It takes initial conditions as a list in form [x1,y1,x2,y2,x3,y3,x4,y4,...], 
    # where x1,y1 are initial conditions for the 1st oscillator
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



t = np.linspace(0, 1000, 1500)

# PARAMETERS
# Every parameter is set by a list, where 0-th element (e.g., omega[0]) represent 
# the value of the parameter for the 0-th oscillator. 
# If the values across the oscillators are the same, you may just put [value]*number_of_oscillators

# Period of different oscillator varies +-1 SND (standart normal distribution) from 24h.
# Omega is derived as (2pi/24+-1SND) 
omeg = [(np.pi*2)/(24+i) for i in np.random.randn(10)]
params = ([0.1]*10,[1]*10,omeg,[0.1]*10,[0.05]*10)
state0 = [1,1]*10

# Solving ODEs
x1 = odeint(oscillator_system, state0, t, args = (params))

n = int(len(state0) / 2)

# Plotting x-coordinates of oscillators, mean and variance
plt.figure(figsize=(20,8))
for i in range(n):
    plt.plot(x1[:, 2*i], label = 'x coord of {}st osc'.format(i))
plt.plot(np.mean(x1, axis=1), 'o', label = 'mean')
#plt.plot(np.var(x1, axis=1), 'o', label = 'variance of coordinate')

plt.ylim(-2,5)
plt.legend()
plt.show()

