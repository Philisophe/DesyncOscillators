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



def analysis (solutions):
    zeroCrossInd = []
    zeroCrossVal = []
    zeroCrossT = []
    period = []
    extrVal = [[] for _ in range(n)] # Let's create n different empty lists
    extrT = [[] for _ in range(n)]
    phases=[]
    
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
        
        # Phases are just all the time-points of zero-crossings, minima and maxima
        phases.append(zeroCrossT[i].tolist()+extrT[i]) 
        phases[i].sort()
    
    return { "zeroCrossInd":zeroCrossInd, "zeroCrossVal":zeroCrossVal, 
            "zeroCrossT":zeroCrossT, "period":period, 
            "extrVal":extrVal, "extrT":extrT, "phases":phases}


#Running average
# x - data, N - window size, N2 - number of runs (if N2=0 - 1 run, if N2=1 - 2 runs etc.)
def running_mean(x, N,N2=0):
    if N2==0:
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    else:
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return running_mean(((cumsum[N:] - cumsum[:-N]) / float(N)),N,N2-1)

def cart2pol(x, y):
    theta = np.rad2deg(np.arctan2(y, x)) # ix1pol = [cart2pol(x1[:,i*2],x1[:,i*2 +1]) for i in range(n)]n degrees
    rho = np.hypot(x, y)
    return [theta, rho]

# Separates x- and y- coordinates
# int(np.shape(solution)[1]/2)) is the same as n
def sep(solution):
    solution_x = [solution[:,i*2] for i in range(int(np.shape(solution)[1]/2))]
    solution_y = [solution[:,i*2+1] for i in range(int(np.shape(solution)[1]/2))]
    return [solution_x,solution_y]

# Translate the result of odeint() function into the polar coordinates 
# and returns only the positive values
def sol2pol(solution):
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

"""
###################################################################################################################################
HERE THE CODE THAT DEFINES THE NECESSARY FUNCTIONS STOPPS
THE REAL SCIENCE STARTS HERE
###################################################################################################################################
"""
# PARAMETERS
# Every parameter is set by a list, where 0-th element (e.g., omega[0]) represent 
# the value of the parameter for the 0-th oscillator. 
# If the values across the oscillators are the same, you may just put [value]*number_of_oscillators

# Period of different oscillator varies +-1 SND (standart normal distribution) from 24h.
# Omega is derived as (2pi/24+-1SND) 
"""
###################################################################################################################
"""


n=2 # Number of oscillators
t = np.linspace(0, 600, 6000)
state0 = [2,2]*n # Initial conditions


omeg = [(np.pi*2)/(24+i) for i in np.random.randn(n)]
params = ([0.1]*n,[1]*n,omeg,[0.1]*n,[0.01]*n) # alpha (amplitude-relaxation rate), amplitude, omega (angular speed), twist, K (coupling strength)


# Solving ODEs
x1 = odeint(oscillator_system, state0, t, args = (params))



# Plotting x-coordinates of oscillators, mean and variance
plt.figure(figsize=(20,8))
for i in range(n):
    plt.plot(t,x1[:, 2*i], 'o', label = 'x coord of {}st osc'.format(i))
plt.plot(t,np.mean(sep(), axis=0), '-', label = 'mean')

plt.ylim(-2,5)
plt.legend()
plt.show()











"""
# Certain stuff to analyze
x1x=[x1[:,i*2] for i in range(n)] # Only x-coordinate
x1y = [x1[:,(i*2)+1] for i in range(n)] # Only y-coordinate

# Cartesian-to-polar conversion
x1pol = [cart2pol(x1[:,i*2],x1[:,i*2 +1]) for i in range(n)] # Same as x1 but in polar coordinates; changes shape of the array
for i in range(n):
    x1pol[i][0] = abs(x1pol[i][0]) # Taking only the absolute values of phases (to avoid jumps from 179 to -179)

# Taking variance of phases (theta's)
thetas1 = [x1pol[i][0] for i in range(n)] # Extract only thetas
var1 = np.var(thetas1, axis=0)

# Getting the envelope of the variance
extrVal=[]
extrT=[]
diff = np.diff(np.sign(np.diff(var1)))
for j in range(len(diff)):
    if diff[j]!=0:
        extrVal.append( (np.mean(var1[j:j+2])) )
        extrT.append( (np.mean(t[j:j+2])) )
# Taking only the positive part of the envelope
extrVal2 = [extrVal[i*2] for i in range(int(len(extrVal)/2))]
extrT2 = [extrT[i*2] for i in range(int(len(extrVal)/2))]


# Analysis of ODEs
als = analysis(x1)
"""



"""
# One attempt to define phase using extrema and 0-crossings
ph = als['phases']
zc = als['zeroCrossT']
ext = als['extrT']

minlen = min([len(i) for i in ph])
minph = [i[0:minlen-1] for i in ph]
"""



"""
# One attempt to define number of oscillators
n = int(len(state0) / 2) # Number of oscillators
"""




"""
# Generating graph1
mean vs. time for 100 oscillators with different sigmas
NO TWIST
NO COUPLING

n = 100 # Number of oscillators
t = np.linspace(0, 500, 5000)
state0 = [2,2]*n

x1 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 0.5*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)))
x2 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)))
x3 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1.5*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)))
x4 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 2*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)))

x1x=[x1[:,i*2] for i in range(n)] # Only x-coordinate
x2x=[x2[:,i*2] for i in range(n)] # Only x-coordinate
x3x=[x3[:,i*2] for i in range(n)] # Only x-coordinate
x4x=[x4[:,i*2] for i in range(n)] # Only x-coordinate

plt.figure(figsize=(20,8))

plt.plot (t, np.mean(x1x,axis=0), label = 'sigma=0.5')
plt.plot (t, np.mean(x2x,axis=0), label = 'sigma=1')
plt.plot (t, np.mean(x3x,axis=0), label = 'sigma=1.5')
plt.plot (t, np.mean(x4x,axis=0), label = 'sigma=2')
plt.ylabel ('Mean of x-coordinate of 100 oscillators')
plt.xlabel ('time, hours')
plt.ylim(-1.5,2.5)
plt.legend()
plt.show()
"""


"""
# Generating graph2
var(x-coordinate) vs. time for 100 oscillators with different sigmas
NO TWIST
NO COUPLING

n = 100 # Number of oscillators
t = np.linspace(0, 500, 5000)
state0 = [2,2]*n

params = ([0.1]*n,[1]*n,omeg,[0.0]*n,[0.0]*n)
x1 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 0.5*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)))
x2 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)))
x3 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1.5*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)))
x4 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 2*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)))

x1x=[x1[:,i*2] for i in range(n)] # Only x-coordinate
x2x=[x2[:,i*2] for i in range(n)] # Only x-coordinate
x3x=[x3[:,i*2] for i in range(n)] # Only x-coordinate
x4x=[x4[:,i*2] for i in range(n)] # Only x-coordinate

plt.figure(figsize=(20,8))

plt.plot (t, np.mean(x1x,axis=0), label = 'sigma=0.5')
plt.plot (t, np.mean(x2x,axis=0), label = 'sigma=1')
plt.plot (t, np.mean(x3x,axis=0), label = 'sigma=1.5')
plt.plot (t, np.mean(x4x,axis=0), label = 'sigma=2')

plt.ylabel ('Variance of x-coordinate of 100 oscillators')
plt.xlabel ('time, hours')

plt.ylim(-1.5,2.5)
plt.legend()
plt.show()
"""


    
    

















