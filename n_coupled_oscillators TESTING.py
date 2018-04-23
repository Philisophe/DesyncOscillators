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



t = np.linspace(0, 1000, 1000)
ideal_period = 24

# PARAMETERS
# Every parameter is set by a list, where 0-th element (e.g., omega[0]) represent 
# the value of the parameter for the 0-th oscillator. 
# If the values across the oscillators are the same, you may just put [value]*number_of_oscillators

# Period of different oscillator varies +-1 SND (standart normal distribution) from 24h.
# Omega is derived as (2pi/24+-1SND) 
omeg = [(np.pi*2)/(24+i) for i in np.random.randn(10)]
params = ([0.1]*10,[1]*10,omeg,[0.1]*10,[0.01]*10)
state0 = [1,1]*10

# Solving ODEs
x1 = odeint(oscillator_system, state0, t, args = (params))

n = int(len(state0) / 2)



#################################
# 18 APRIL WORK #
# ANALYSIS OF PHASES
#################################

    
def analysis (solutions):
    zeroCrossInd = []
    zeroCrossVal = []
    zeroCrossT = []
    period = []
    extrVal = [[] for _ in range(n)] # Let's create n different empty lists
    extrT = [[] for _ in range(n)]

    
    for i in range(n):
        zeroCrossInd.append(np.where(np.diff(np.sign(solutions[:,i*2])))[0])
        # Going through only even columns of solutions array (which contain the x-coordinates)
        # The crossing happens between indices in zeroCrossInd and next one. E.g., zeroCrossInd[0][0] (the 0th index for the 1st oscillator) is 5, meaning that 0 is somewhere between 5th and 6th timepoint.
                
        # the values themselves, should be really close to 0
        zeroCrossVal.append((solutions[:,i*2][zeroCrossInd[i]] + solutions[:,i*2][zeroCrossInd[i]+1])/2)
        
        
        #the approximate times of actual crossings
        zeroCrossT.append( (t[zeroCrossInd[i]]+t[zeroCrossInd[i]+1])/2 ) # takes the time in between two x-values of opposing signs
        
        period.append( np.diff(zeroCrossT[i]) )
        
        # Looking for local maxima, minima
        # When result of np.diff() changes the sign - it's when the max of min occured 
        # diff changes it's meaning after every iteration
        diff = np.diff(np.sign(np.diff(solutions[:,i*2])))
        for j in range(len(diff)):
            if diff[j]!=0:
                extrVal[i].append( (np.mean(solutions[:,i*2][j:j+2])) )
                extrT[i].append( (np.mean(t[j:j+2])) )
        
        
    return { "zeroCrossInd":zeroCrossInd, "zeroCrossVal":zeroCrossVal, 
            "zeroCrossT":zeroCrossT, "period":period, 
            "extrVal":extrVal, "extrT":extrT}

als = analysis(x1)
#ph = als['phases']
zc = als['zeroCrossT']
ext = als['extrT']



def mlen(x):
    return len(min(x,key=len))

"""
#for i in minlen(zc+ext):
#    if (zeroCrossT[i]
#    phases.append(zeroCrossT[i].tolist())
zcdif = np.diff(zc[:,mlen(zc)],axis=0)
phases = np.zeros((n,mlen(zc)+mlen(ext)))

for i in range(np.shape(zcdif)[0]):
    for j in range(np.shape(zcdif)[1]):
        if zcdif[i][j]<=12:
            phases[i][j]=zc[i][j]
        else:
            phases[i][j]=zc[i][j+1]
THAT'S REALLY UNEFFICIENT, PLEASE DON'T GO THIS WAY
"""

analyt_sig=np.zeros()


#minlen = min([len(i) for i in ph])
#m = mlen(phases)
#minph = [i[0:m-1] for i in ph]


# Plotting x-coordinates of oscillators, mean and variance
plt.figure(figsize=(20,8))
for i in range(n):
    plt.plot(t,x1[:, 2*i], 'o', label = 'x coord of {}st osc'.format(i))
plt.plot(t,np.mean(x1, axis=1), '+', label = 'mean') # WRONG MEAN, takes into account also y-coordinate
#plt.plot(anls[])
#plt.plot(np.var(x1, axis=1), 'o', label = 'variance of coordinate')

plt.ylim(-2,5)
plt.legend()
plt.show()
