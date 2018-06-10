# This code integrates over a system of ODEs of coupled oscillators with parameters.
# The single oscillator is implemented as a modified Poincare oscillator (with twist - parameter, linking period and amplitude).


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





def ode_rand3(number_of_oscillators, timepoints, state0, params, randMulti):   
    """The function models the behaviour of system of coupled Poincare oscillators with noise. 
    To do that, it executes odeint() function with oscillator_system as a first parameter multiple times in a row, changing each time noisy variable E to a random value drawn from standart normal distribution (SND).
    The dispersion (sigma) of SND for E is set by randMulti parameter.
    
    timepoints - expects a np.linspace(x,y,z) so that for every (x-y)=0.5 there would be at least z=10.
    In other words, the stop should be made every 0.5h and the should be exactly 10 datapoint per execution.
    
    The function returns solutions in the form of np.array.
    
    Example of execution: 
    2 oscillator system executed 160 consequetive times with 10 datapoints each, starting from [2,2] and [3,3] with params as all parameters except for noise, which is set explicitely by E.
    
    x4=ode_rand2(2,160,np.linspace(0,0.5,10),[2,2,3,3],params,0.1)
    plt.plot(x4[:,0], label = 'x-coordinate of the 1st oscillator')
    plt.legend()
    
    """
    t = timepoints # Shortcut
    n = number_of_oscillators # Shortcut
    iterations = int(len(timepoints)/10) # because we decided that the stop should be made every 0.5h, 
    # and that there should be at least 10 dp for every 0.5h, that means that number of stops is exactly number of datapoints/10.
    solutions = np.zeros((len(t)-iterations+1,n*2)) # Creates array of zeros of an appropriate size to store iterative executions of odeint() function
    
    time = [] # Variable for storing the effective timepoints (in case I lose any)
    start=0
    
    for i in range(iterations):
        
        E = randMulti*np.random.randn(n) # Creates vector of random numbers from SND
        
        s = odeint(oscillator_system, state0, t[start:start+10], args = ((params[0], params[1], params[2], params[3], params[4], E))) # The parameters: alpha, amplitude, omega, twist, coupling
        solutions[start:start+10] = s
        
        state0 = s[-1].tolist()
        time.append(t[start:start+10])
        start = start+10-1
        
        #print ('s: ', s ,'\n')
        #print ('solutions: ', solutions)
        #print('start: ', start)
        #print ('time: ', time)

    return remdup(flat_list(time)), solutions






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
    """Finds the extrema of the function. Returns [timepoints of extrema, values of extrema] list.
    
    Very poorly deals with data on the borders"""
    diff = np.diff(np.sign(np.diff(x)))
    extrT=[]
    extrVal=[]
    for i in range(len(diff)):
        if diff[i]!=0:
            extrVal.append(np.mean(x[i:i+2]))
            extrT.append(np.mean(t[i:i+2]))
    return [extrT,extrVal]


def maxs(list_extr):
    #Returns every odd element. Designed to be used in combination with extr(), e.g. maxs(extr(solutions))
    maxsV=[]
    maxsT=[]
    for i in range(int(len(list_extr[0])/2)):
        maxsV.append(list_extr[1][i*2 + 1])
        maxsT.append(list_extr[0][i*2 + 1])
    return [maxsT,maxsV]


def maxs2(list_extr):
    """Returns maxima in the form [values,timepoints]. 
    Designed to be used in combination with extr(), e.g. maxs(extr(solutions)).
    
    Use only with oscillatory functions. Expects input to be list
    
    np.argpartition() works not like you expect to! """
    
        
    l = int(len(list_extr[0])/2) # l is total number of extrema/2. In other words, for even number of extrema its number of maxima or minima.
    ind = np.argpartition(list_extr[1], -l)[-l:] # Extract indices of the first "l" extrema (they will be maxima) out of the extrVal
    return [np.array(list_extr[0])[ind], np.array(list_extr[1])[ind]]


def maxs3(list_extr):
    """Returns maxima in the form [values,timepoints]. 
    Takes every odd element. Designed to be used in combination with extr(), e.g. maxs(extr(solutions))
    
    Adds the initial conditions t=0 and x(t=0)=1 in the beginning of the the lists
    VERY CRUDE METHOD, use only with oscillatory functions, starting with initial conditions [1,0] at time t=0."""
    maxsV=[]
    maxsT=[]
    for i in range(int(len(list_extr[0])/2)):
        maxsV.append(list_extr[1][i*2 + 1])
        maxsT.append(list_extr[0][i*2 + 1])
    maxsT.insert(0,0) # But I can hardware it as 0
    maxsV.insert(0,1)
    return [maxsT,maxsV]

def maxs4 (data):
    """Takes the data, returns the indices of the maxima in value.
    
    Let's find maxima of variances in the data:
    v1 = np.var(x1x, axis=0)
    plt.plot(time, v1)
    plt.plot(time[maxs4(v1)], v1[maxs4(v1)])"""
    a = (np.diff(np.sign(np.diff(data)))<0).nonzero()[0]+1
    return a


def me(x):
    return maxs(extr(x))

def me2(x):
    """Returns maxs(extr(np.mean(x,axis=0))).
    In other words - maxima of mean of x as a list [times, values]"""
    return me(np.mean(x, axis=0))

def me3(x, N, N2):
    return me(run_mean(np.mean(x, axis=0),N,N2))

def me4(x):
    return maxs3(extr(x))

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
    """Computes phase (theta) variance
    Returns list [phase variance, the upper envelope of phase variance as maxima]
    
    Looks ugly, but that's OK; 
    relies on 't'"""
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
    """Returns ax+b"""
    return (a*x + b)

def quad(x, a, b, c):
    """Returns ax^2 + bx + c"""
    return (a*(x**2) + b*x + c)

def cub(x,a,b,c,d):
    """Returns ax^3 + bx^2 + cx + d"""
    return (a*(x**3) + b*(x**2) + c*x + d)

def expon(x, a, b, c):
    """Returns ae^(-bx) + c"""
    return a * np.exp(-b * x) + c

def roundl(x, n):
    """Round list x up to n digits"""
    return list(map(lambda x,n: round(x,n), x,[n]*len(x)))

def norm(x):
    """Returns normalised array or list"""
    return x/np.mean(x) 

def areeq(x,y):
    """Says if two arrays are equal. If one is bigger than other - shows answers only for the number of elements from the smaller one."""
    return list(map(lambda x,y:x==y, x,y))

def slp2ang(x):
    """Converts slope of the linear function to angles"""
    return list(map(lambda x:np.rad2deg(np.arctan(x)),x))

def r_sq(function, xdata, ydata, popt):
    """Computes R^2 statistical metric using 'popt' parameter from scipy.optimize.curve_fit() function.
    The formula it uses is:
        r_squared = 1 - (residual sum of squares/total sum of squares)"""
    residuals = ydata - function(xdata,*popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    return (1-(ss_res/ss_tot))

def per(xnx, limit):
    """Function for finding the periods.
    Input: x1x[0]-type data (only for 1 specific oscillator)
    Output: np.array of periods for a given oscillator after "limit"
    Uses maxs4() to extract the maxima."""
    
    d = t[maxs4(xnx)] # timepoint values with maxima on them
    d = d[np.where(d>limit)]
    return np.diff(d)

def pers(xnx, limit):
    """Function for finding the periods for the whole dataset.
    Input: data in x1x-type
    Averages periods of sigular oscillators.
    Ouput: [mean_periods of the given xnx file]
    """
    mean_periods=[]
    for i in range(len(xnx)):
        d = t[maxs4(xnx[i])] # timepoint values with maxima on them
        d = d[np.where(d>limit)] # Take only timepoints values after limit hours
        mean_periods.append(np.mean(np.diff(d))) # np.diff(d) - gives you periods; # Appends mean periods (1st entry - mean period of the 1st oscillator)
    return mean_periods

    

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


"""
# EXAMPLE OF SETTING THE SYSTEM UP

n=2 # Number of oscillators
t = np.linspace(0, 600, 6000)
state0 = [2,2]*n # Initial conditions


omeg = [(np.pi*2)/(24+i) for i in np.random.randn(n)]
params = ([0.1]*n,[1]*n,omeg,[0.1]*n,[0.01]*n) # alpha (amplitude-relaxation rate), amplitude, omega (angular speed), twist, K (coupling strength)


# Solving ODEs
#x1 = odeint(oscillator_system, state0, t, args = (params))



# Plotting x-coordinates of oscillators, mean and variance
plt.figure(figsize=(20,8))
for i in range(n):
    plt.plot(t,x1[:, 2*i], 'o', label = 'x coord of {}st osc'.format(i))
#plt.plot(t,np.mean(sep(), axis=0), '-', label = 'mean')

plt.ylim(-2,5)
plt.legend()
plt.show()
"""


"""
# One attempt to define phase using extrema and 0-crossings
# Analysis of ODEs
als = analysis(x1)
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
##############################
# STARTING THE ENGINE
#############################
n = 1000
t = np.linspace(0, 400, 4000)
state0 = [1,0]*n

x1 = np.load("/home/kalashnikov/Code/Variables for my code/Heterogeneity/Other state0/1000 oscillators with sigma 0.5 and state0 [1,0].npy")
x2 = np.load("/home/kalashnikov/Code/Variables for my code/Heterogeneity/Other state0/1000 oscillators with sigma 1 and state0 [1,0].npy")
x3 = np.load("/home/kalashnikov/Code/Variables for my code/Heterogeneity/Other state0/1000 oscillators with sigma 1.5 and state0 [1,0].npy")
x4 = np.load("/home/kalashnikov/Code/Variables for my code/Heterogeneity/Other state0/1000 oscillators with sigma 2 and state0 [1,0].npy")

x1x = sep(x1)[0]
x2x = sep(x2)[0]
x3x = sep(x3)[0]
x4x = sep(x4)[0]


"""






"""
##########################################################################################################
##########################################################################################################

# THIS IS DATA ONLY FOR NON-NOISY SYSTEMS

####


# GRAPH 1

# ONE NUMBER OF OSCILLATORS; MANY SIGMAS

########################################
MEAN(x-coordinate)
#######################################

#NO TWIST
#NO COUPLING

n = 1000 # Number of oscillators
t = np.linspace(0, 370, 3700)
state0 = [2,2]*n

x1 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 0.5*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)))
x2 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)))
x3 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1.5*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)))
x4 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 2*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)))

x1x = sep(x1)[0]
x2x = sep(x2)[0]
x3x = sep(x3)[0]
x4x = sep(x4)[0]

plt.figure(figsize=(20,8))

plt.plot (t, np.mean(x1x,axis=0), label = 'sigma=0.5')
plt.plot (t, np.mean(x2x,axis=0), label = 'sigma=1')
plt.plot (t, np.mean(x3x,axis=0), label = 'sigma=1.5')
plt.plot (t, np.mean(x4x,axis=0), label = 'sigma=2')

plt.ylabel ('Mean of x-coordinate of 1000 oscillators')
plt.xlabel ('time, hours')
plt.ylim(-1.5,2.5)
plt.legend()
plt.show()



############
# Testing me() function
plt.plot(t,np.mean(x1x,axis=0), '--', label="mean")
plt.plot(extr(np.mean(x1x,axis=0))[0],extr(np.mean(x1x,axis=0))[1],'o', label="extrema")
plt.plot(me(np.mean(x1x,axis=0))[0], me(np.mean(x1x,axis=0))[1], "+", label="maxima")
plt.legend()

############
 ONLY MAXIMA FROM MEAN(X)
###########
plt.figure(figsize=(20,8))
plt.plot(t,np.mean(x1x,axis=0), label="sigma 0.5")
plt.plot(t,np.mean(x2x,axis=0), label="sigma 1.0")
plt.plot(t,np.mean(x3x,axis=0), label="sigma 1.5")
plt.plot(t,np.mean(x4x,axis=0), label="sigma 2.0")

plt.plot(me(np.mean(x1x,axis=0))[0], me(np.mean(x1x,axis=0))[1], label="maxima sigma 0.5")
plt.plot(me(np.mean(x2x,axis=0))[0], me(np.mean(x2x,axis=0))[1], label="maxima sigma 1.0")
plt.plot(me(np.mean(x3x,axis=0))[0], me(np.mean(x3x,axis=0))[1], "+", label="maxima sigma 1.5")
plt.plot(me(np.mean(x4x,axis=0))[0], me(np.mean(x4x,axis=0))[1], "+", label="maxima sigma 2.0")

plt.legend()

#############
FITTING TO THE LINEAR
##############

xdata1 = np.array(me(np.mean(x1x, axis=0))[0])
ydata1 = np.array(me(np.mean(x1x, axis=0))[1])

xdata2 = np.array(me(np.mean(x2x, axis=0))[0])
ydata2 = np.array(me(np.mean(x2x, axis=0))[1])

xdata3 = np.array(me(np.mean(x3x, axis=0))[0])
ydata3 = np.array(me(np.mean(x3x, axis=0))[1])

xdata4 = np.array(me(np.mean(x4x, axis=0))[0])
ydata4 = np.array(me(np.mean(x4x, axis=0))[1])
#xydata = [xdata1,ydata1,xdata2,ydata2,xdata3,ydata3,xdata4,ydata4]  
  

popt1,pcov1 = curve_fit(lin,xdata1,ydata1)
popt2,pcov2 = curve_fit(lin,xdata2[0:5],ydata2[0:5])
popt3,pcov3 = curve_fit(lin,xdata3[0:4],ydata3[0:4])
popt4,pcov4 = curve_fit(lin,xdata4[0:3],ydata4[0:3])

tangents = [popt1[0],popt2[0],popt3[0], popt4[0]]
nt = norm(tangents)
rt = roundl(tangents,4)
angles = slp2ang(tangents)

plt.figure(figsize=(10,6))
plt.plot(xdata1,ydata1, 'ro', label = 's=0.5 maxima')
plt.plot(xdata1, lin(xdata1, *popt1), 'r--', label = 's=0.5 fit')

plt.plot(xdata2,ydata2, 'mo', label = 's=1.0 maxima')
plt.plot(xdata2, lin(xdata2, *popt2), 'm--', label = 's=1.0 fit')

plt.plot(xdata3,ydata3, 'bo', label = 's=1.5 maxima')
plt.plot(xdata3, lin(xdata3, *popt3),'b--', label = 's=1.5 fit')

plt.plot(xdata4,ydata4, 'ko', label = 's=2.0 maxima')
plt.plot(xdata4, lin(xdata4, *popt4), 'k--', label = 's=2.0 fit')

# These things need to be applied after xlim() and ylim()
#plt.text(70,1.1,str(rt[0]), rotation=np.rad2deg(np.arctan(rt)))
#plt.text(70,1.03,str(rt[0]) + ' incorrect rotation', rotation=angles[0])

plt.ylabel ('Maxima of mean(x-coordinate) of 1000 oscillators fitted to line')
plt.xlabel ('time, hours')

texttang = ''
for i in rt:
    texttang = texttang+str(i)+' : '
texttang = texttang[:-3]
plt.text(133,1.1,'The slopes of curves are\n' + texttang)

plt.xlim(15,200)
plt.ylim(-0.4,1.3)
plt.legend()


####
Practicing with POLYNOMIALS

data3poly2 = np.polyfit(xdata3,ydata3,2) # Fitting data3 to polynomial of the 2nd degree
#Using poly1d(coefficients)(x-values) you can get y-values
plt.plot(t,np.poly1d(data3poly2)(t), label='deg 2 fit for data3')
plt.plot(xdata3,ydata3, label = 'data 3')
plt.legend()

####


############################################### OR
VAR (x-coordinate)
################################################

plt.figure(figsize=(20,8))

plt.plot (t, np.var(x1x,axis=0), label = 'sigma=0.5')
plt.plot (t, np.var(x2x,axis=0), label = 'sigma=1')
plt.plot (t, np.var(x3x,axis=0), label = 'sigma=1.5')
plt.plot (t, np.var(x4x,axis=0), label = 'sigma=2')

plt.ylabel ('Variance of x-coordinate of 1000 oscillators')
plt.xlabel ('time, hours')

plt.ylim(-1.5,2.5)
plt.legend()
plt.show()



############## OR 
running_mean(VAR (x-coordinate))
##############

plt.figure(figsize=(20,8))

plt.plot (t[:3487], run_mean(np.var(x1x,axis=0),72,2), label = 'sigma=0.5')
plt.plot (t[:3487], run_mean(np.var(x2x,axis=0),72,2), label = 'sigma=1')
plt.plot (t[:3487], run_mean(np.var(x3x,axis=0),72,2), label = 'sigma=1.5')
plt.plot (t[:3487], run_mean(np.var(x4x,axis=0),72,2), label = 'sigma=2')

plt.ylabel ('Variance of x-coordinate of 1000 oscillators with running average (72,3)')
plt.xlabel ('time, hours')

plt.ylim(-1.5,2.5)
plt.legend()
plt.show()

############
plt.figure(figsize=(20,8))

plt.plot (t[:3416], run_mean(np.var(x1x,axis=0),72,3), label = 'sigma=0.5')
plt.plot (t[:3416], run_mean(np.var(x2x,axis=0),72,3), label = 'sigma=1')
plt.plot (t[:3416], run_mean(np.var(x3x,axis=0),72,3), label = 'sigma=1.5')
plt.plot (t[:3416], run_mean(np.var(x4x,axis=0),72,3), label = 'sigma=2')

plt.ylabel ('Variance of x-coordinate of 1000 oscillators with running average (72,4)')
plt.xlabel ('time, hours')

plt.legend()
plt.show()

########### 
# 12h as a window
plt.figure(figsize=(20,8))

plt.plot (t[:3762], run_mean(np.var(x1x,axis=0),120,1), label = 'sigma=0.5')
plt.plot (t[:3762], run_mean(np.var(x2x,axis=0),120,1), label = 'sigma=1')
plt.plot (t[:3762], run_mean(np.var(x3x,axis=0),120,1), label = 'sigma=1.5')
plt.plot (t[:3762], run_mean(np.var(x4x,axis=0),120,1), label = 'sigma=2')

plt.ylabel ('Variance of x-coordinate of 1000 oscillators with running average (72,4)')
plt.xlabel ('time, hours')

plt.legend()
plt.show()


#############
FITTING TO THE DIFFERENT FUNCTIONS / PLOTTING IN LOG()
##############
#### Plotting in log-scale
plt.plot (t[:3416], run_mean(np.var(x1x,axis=0),72,3), label = 'sigma=0.5')
plt.plot (t[:3416], run_mean(np.var(x2x,axis=0),72,3), label = 'sigma=1')
plt.plot (t[:3416], run_mean(np.var(x3x,axis=0),72,3), label = 'sigma=1.5')
plt.plot (t[:3416], run_mean(np.var(x4x,axis=0),72,3), label = 'sigma=2')

plt.ylabel ('Variance of x-coordinate of 1000 oscillators with running average (72,4)')
plt.xlabel ('time, hours')
plt.yscale('log')
plt.legend()
plt.show()


###### Fitting to quadratic

xdata=t[:3416]
ydata1=run_mean(np.var(x1x,axis=0),72,3)
ydata2=run_mean(np.var(x2x,axis=0),72,3)
ydata3=run_mean(np.var(x3x,axis=0),72,3)
ydata4=run_mean(np.var(x4x,axis=0),72,3)

popt1,pcov1 = curve_fit(quad,xdata,ydata1)
popt2,pcov2 = curve_fit(quad,xdata,ydata2)
popt3,pcov3 = curve_fit(quad,xdata,ydata3)
popt4,pcov4 = curve_fit(quad,xdata,ydata4)


plt.figure(figsize=(20,8))

plt.plot (xdata, ydata1, 'r--',label = 'sigma=0.5 data')
plt.plot(xdata,quad(xdata,*popt1),'r-', label = 'fit')

plt.plot (xdata, ydata2,'m--', label = 'sigma=1')
plt.plot(xdata,quad(xdata,*popt2),'m-', label = 'fit')

plt.plot (xdata, ydata3,'b--', label = 'sigma=1.5')
plt.plot(xdata,quad(xdata,*popt3),'b-', label = 'fit')

plt.plot (xdata, ydata4, 'k--', label = 'sigma=2')
plt.plot(xdata,quad(xdata,*popt4),'k-', label = 'fit')

plt.ylabel ('Variance of x-coordinate of 1000 oscillators with running average (72,4) fitted to quadratic function')
plt.xlabel ('time, hours')
#plt.yscale('log')
plt.legend()
plt.show()

###### Fitting to quadratic - better fits

xdata=t[:3416]
ydata1=run_mean(np.var(x1x,axis=0),72,3)
ydata2=run_mean(np.var(x2x,axis=0),72,3)
ydata3=run_mean(np.var(x3x,axis=0),72,3)
ydata4=run_mean(np.var(x4x,axis=0),72,3)

popt1,pcov1 = curve_fit(quad,xdata[:1100],ydata1[:1100])
popt2,pcov2 = curve_fit(quad,xdata[:900],ydata2[:900])
popt3,pcov3 = curve_fit(quad,xdata[:400],ydata3[:400])
popt4,pcov4 = curve_fit(quad,xdata[:300],ydata4[:300])

plt.figure(figsize=(16,8))

plt.plot (xdata, ydata1, 'r--',label = 'sigma=0.5 data')
plt.plot(xdata,quad(xdata,*popt1),'r-', label = 'fit')
plt.plot (xdata, ydata2,'m--', label = 'sigma=1')
plt.plot(xdata,quad(xdata,*popt2),'m-', label = 'fit')
plt.plot (xdata, ydata3,'b--', label = 'sigma=1.5')
plt.plot(xdata,quad(xdata,*popt3),'b-', label = 'fit')
plt.plot (xdata, ydata4, 'k--', label = 'sigma=2')
plt.plot(xdata,quad(xdata,*popt4),'k-', label = 'fit')

plt.ylabel ('Variance of x-coordinate of 1000 oscillators with running average (72,4) fitted to quadratic function')
plt.xlabel ('time, hours')
#plt.xlim(-,)
plt.ylim(-0.05,0.55)
#plt.yscale('log')
plt.legend()
plt.show()

###########  Fitting to quadratic - better fits and other running_average

xdata=t[:3462]
ydata1=run_mean(np.var(x1x,axis=0),120,1)
ydata2=run_mean(np.var(x2x,axis=0),120,1)
ydata3=run_mean(np.var(x3x,axis=0),120,1)
ydata4=run_mean(np.var(x4x,axis=0),120,1)

popt1,pcov1 = curve_fit(quad,xdata[:1000],ydata1[:1000])
popt2,pcov2 = curve_fit(quad,xdata[:500],ydata2[:500])
popt3,pcov3 = curve_fit(quad,xdata[:300],ydata3[:300])
popt4,pcov4 = curve_fit(quad,xdata[:200],ydata4[:200])

coefficients = [popt1[0],popt2[0],popt3[0], popt4[0]]
ct = roundl(coefficients,6)

plt.figure(figsize=(12,10))
plt.plot (xdata, ydata1, 'r--',label = 'sigma=0.5 data')
plt.plot(xdata,quad(xdata,*popt1),'r-', label = 'fit')
plt.plot (xdata, ydata2,'m--', label = 'sigma=1')
plt.plot(xdata,quad(xdata,*popt2),'m-', label = 'fit')
plt.plot (xdata, ydata3,'b--', label = 'sigma=1.5')
plt.plot(xdata,quad(xdata,*popt3),'b-', label = 'fit')
plt.plot (xdata, ydata4, 'k--', label = 'sigma=2')
plt.plot(xdata,quad(xdata,*popt4),'k-', label = 'fit')

plt.ylabel ('Variance of x-coordinate of 1000 oscillators with running average (12h,2) fitted to quadratic function')
plt.xlabel ('time, hours')
plt.ylim(-0.05,0.55)
plt.xlim(-5,150)

texttang = ''
for i in ct:
    texttang = texttang+str(i)+' : '
texttang = texttang[:-3]
plt.text(100,0,'The first coefficients of curves are\n' + texttang)
#plt.yscale('log')
plt.legend()
plt.show()

####### Fitting to the exponential

####
xdata=t[:3462]
ydata1=run_mean(np.var(x1x,axis=0),120,1)
ydata2=run_mean(np.var(x2x,axis=0),120,1)
ydata3=run_mean(np.var(x3x,axis=0),120,1)
ydata4=run_mean(np.var(x4x,axis=0),120,1)
###
popt1,pcov1 = curve_fit(expon,xdata,ydata1)
popt2,pcov2 = curve_fit(expon,xdata,ydata2)
popt3,pcov3 = curve_fit(expon,xdata,ydata3)
popt4,pcov4 = curve_fit(expon,xdata,ydata4)

plt.figure(figsize=(12,8))

plt.plot (xdata, ydata1, 'r--',label = 'sigma=0.5 data')
plt.plot(xdata,expon(xdata,*popt1),'r-', label = 'fit')

plt.plot (xdata, ydata2,'m--', label = 'sigma=1')
plt.plot(xdata,expon(xdata,*popt2),'m-', label = 'fit')

plt.plot (xdata, ydata3,'b--', label = 'sigma=1.5')
plt.plot(xdata,expon(xdata,*popt3),'b-', label = 'fit')

plt.plot (xdata, ydata4, 'k--', label = 'sigma=2')
plt.plot(xdata,expon(xdata,*popt4),'k-', label = 'fit')

plt.ylabel ('Variance of x-coordinate of 1000 oscillators with running average (72,4) fitted to exponential function')
plt.xlabel ('time, hours')
#plt.yscale('log')
plt.xlim(-5,150)
plt.legend()
plt.show()



############################################### OR
VAR (phase)
##############################################

plt.figure(figsize=(20,8))
plt.plot (t, phvar(x1)[0], label = 'sigma=0.5')
plt.plot (t, phvar(x2)[0], label = 'sigma=1')
plt.plot (t, phvar(x3)[0], label = 'sigma=1.5')
plt.plot (t, phvar(x4)[0], label = 'sigma=2')

plt.ylabel ('Variance of phase of 1000 oscillators')
plt.xlabel ('time, hours')

plt.legend()
plt.show()

################# OR
running_mean(var(phase))
##################
plt.figure(figsize=(20,8))
plt.plot (t[:3487], run_mean(phvar(x1)[0],72,2), label = 'sigma=0.5')
plt.plot (t[:3487], run_mean(phvar(x2)[0],72,2), label = 'sigma=1')
plt.plot (t[:3487], run_mean(phvar(x3)[0],72,2), label = 'sigma=1.5')
plt.plot (t[:3487], run_mean(phvar(x4)[0],72,2), label = 'sigma=2')

plt.ylabel ('Variance of phase of 1000 oscillators with running average (72,3)')
plt.xlabel ('time, hours')

plt.legend()
plt.show()

###########
plt.figure(figsize=(20,8))
plt.plot (t[:3416], run_mean(phvar(x1)[0],72,3), label = 'sigma=0.5')
plt.plot (t[:3416], run_mean(phvar(x2)[0],72,3), label = 'sigma=1')
plt.plot (t[:3416], run_mean(phvar(x3)[0],72,3), label = 'sigma=1.5')
plt.plot (t[:3416], run_mean(phvar(x4)[0],72,3), label = 'sigma=2')

plt.ylabel ('Variance of phase of 1000 oscillators with running average (72,4)')
plt.xlabel ('time, hours')
#plt.yscale('log')
#plt.xscale('log')

plt.legend()
plt.show()

#########
plt.figure(figsize=(16,8))
plt.plot (t[:3462], run_mean(phvar(x1)[0],120,1), label = 'sigma=0.5')
plt.plot (t[:3462], run_mean(phvar(x2)[0],120,1), label = 'sigma=1')
plt.plot (t[:3462], run_mean(phvar(x3)[0],120,1), label = 'sigma=1.5')
plt.plot (t[:3462], run_mean(phvar(x4)[0],120,1), label = 'sigma=2')

plt.ylabel ('Variance of phase of 1000 oscillators with running average (120,2)')
plt.xlabel ('time, hours')
#plt.yscale('log')
#plt.xscale('log')

plt.legend()
plt.show()

###########
FITTING TO:
###########
LINEAR FUNCTION

xdata=t[:3416]
ydata1=run_mean(phvar(x1)[0],72,3)
ydata2=run_mean(phvar(x2)[0],72,3)
ydata3=run_mean(phvar(x3)[0],72,3)
ydata4=run_mean(phvar(x4)[0],72,3)

popt1,pcov1 = curve_fit(lin,xdata,ydata1)
popt2,pcov2 = curve_fit(lin,xdata,ydata2)
popt3,pcov3 = curve_fit(lin,xdata,ydata3)
popt4,pcov4 = curve_fit(lin,xdata,ydata4)


plt.figure(figsize=(16,8))

plt.plot (xdata, ydata1, 'r--',label = 'sigma=0.5 data')
plt.plot(xdata,lin(xdata,*popt1),'r-', label = 'fit')

plt.plot (xdata, ydata2,'m--', label = 'sigma=1')
plt.plot(xdata,lin(xdata,*popt2),'m-', label = 'fit')

plt.plot (xdata, ydata3,'b--', label = 'sigma=1.5')
plt.plot(xdata,lin(xdata,*popt3),'b-', label = 'fit')

plt.plot (xdata, ydata4, 'k--', label = 'sigma=2')
plt.plot(xdata,lin(xdata,*popt4),'k-', label = 'fit')

plt.ylabel ('Variance of phase of 1000 oscillators with running average (72,4) fitted to linear function')
plt.xlabel ('time, hours')
#plt.yscale('log')
#plt.xscale('log')

plt.legend()
plt.show()


###########
 window(120,1) - better fit
######
xdata=t[:3462]
ydata1=run_mean(phvar(x1)[0],120,1)
ydata2=run_mean(phvar(x2)[0],120,1)
ydata3=run_mean(phvar(x3)[0],120,1)
ydata4=run_mean(phvar(x4)[0],120,1)

popt1,pcov1 = curve_fit(lin,xdata[40:2000],ydata1[40:2000])
popt2,pcov2 = curve_fit(lin,xdata[:800],ydata2[:800])
popt3,pcov3 = curve_fit(lin,xdata[:700],ydata3[:700])
popt4,pcov4 = curve_fit(lin,xdata[:500],ydata4[:500])

plt.figure(figsize=(16,8))

plt.plot (xdata, ydata1, 'r--',label = 'sigma=0.5 data')
plt.plot(xdata,lin(xdata,*popt1),'r-', label = 'fit')

plt.plot (xdata, ydata2,'m--', label = 'sigma=1')
plt.plot(xdata,lin(xdata,*popt2),'m-', label = 'fit')

plt.plot (xdata, ydata3,'b--', label = 'sigma=1.5')
plt.plot(xdata,lin(xdata,*popt3),'b-', label = 'fit')

plt.plot (xdata, ydata4, 'k--', label = 'sigma=2')
plt.plot(xdata,lin(xdata,*popt4),'k-', label = 'fit')

plt.ylabel ('Variance of phase of 1000 oscillators with running average (120,2) fitted to linear function')
plt.xlabel ('time, hours')
plt.ylim(-50,3000)
#plt.yscale('log')
#plt.xscale('log')

plt.legend()
plt.show()

###################
QUADRATIC

###
popt1,pcov1 = curve_fit(quad,xdata[:1500],ydata1[:1500])
popt2,pcov2 = curve_fit(quad,xdata[:800],ydata2[:800])
popt3,pcov3 = curve_fit(quad,xdata[:600],ydata3[:600])
popt4,pcov4 = curve_fit(quad,xdata[:500],ydata4[:500])
###


plt.figure(figsize=(16,8))

plt.plot (xdata, ydata1, 'r--',label = 'sigma=0.5 data')
plt.plot(xdata,quad(xdata,*popt1),'r-', label = 'fit')

plt.plot (xdata, ydata2,'m--', label = 'sigma=1')
plt.plot(xdata,quad(xdata,*popt2),'m-', label = 'fit')

plt.plot (xdata, ydata3,'b--', label = 'sigma=1.5')
plt.plot(xdata,quad(xdata,*popt3),'b-', label = 'fit')

plt.plot (xdata, ydata4, 'k--', label = 'sigma=2')
plt.plot(xdata,quad(xdata,*popt4),'k-', label = 'fit')

plt.ylabel ('Variance of phase of 1000 oscillators with running average (72,4) fitted to quadratic function')
plt.xlabel ('time, hours')
#plt.yscale('log')
#plt.xscale('log')

plt.legend()
plt.show()

####### 
 window(120,1) - better fit
######
popt1,pcov1 = curve_fit(quad,xdata[:800],ydata1[:800])
popt2,pcov2 = curve_fit(quad,xdata[:400],ydata2[:400])
popt3,pcov3 = curve_fit(quad,xdata[:300],ydata3[:300])
popt4,pcov4 = curve_fit(quad,xdata[:200],ydata4[:200])

coefficients = [popt1[0],popt2[0],popt3[0], popt4[0]]
ct = roundl(coefficients,4)

plt.figure(figsize=(10,8))

plt.plot (xdata, ydata1, 'r--',label = 'sigma=0.5 data')
plt.plot(xdata,quad(xdata,*popt1),'r-', label = 'fit')

plt.plot (xdata, ydata2,'m--', label = 'sigma=1')
plt.plot(xdata,quad(xdata,*popt2),'m-', label = 'fit')

plt.plot (xdata, ydata3,'b--', label = 'sigma=1.5')
plt.plot(xdata,quad(xdata,*popt3),'b-', label = 'fit')

plt.plot (xdata, ydata4, 'k--', label = 'sigma=2')
plt.plot(xdata,quad(xdata,*popt4),'k-', label = 'fit')

plt.ylabel ('Variance of phase of 1000 oscillators with running average (120,2) fitted to quadratic function')
plt.xlabel ('time, hours')
plt.ylim(-50,3000)
#plt.yscale('log')
#plt.xscale('log')
plt.xlim(-5,150)

texttang = ''
for i in ct:
    texttang = texttang+str(i)+' : '
texttang = texttang[:-3]
plt.text(100,50,'The coefficients of curves are\n' + texttang)

plt.legend()
plt.show()


############
EXPONENTIAL

DOESN'T WORK -RuntimeError - overflow

popt1,pcov1 = curve_fit(expon,xdata,ydata1)
popt2,pcov2 = curve_fit(expon,xdata,ydata2)
popt3,pcov3 = curve_fit(expon,xdata,ydata3)
popt4,pcov4 = curve_fit(expon,xdata,ydata4)


plt.figure(figsize=(16,8))

plt.plot (xdata, ydata1, 'r--',label = 'sigma=0.5 data')
plt.plot(xdata,expon(xdata,*popt1),'r-', label = 'fit')

plt.plot (xdata, ydata2,'m--', label = 'sigma=1')
plt.plot(xdata,expon(xdata,*popt2),'m-', label = 'fit')

plt.plot (xdata, ydata3,'b--', label = 'sigma=1.5')
plt.plot(xdata,expon(xdata,*popt3),'b-', label = 'fit')

plt.plot (xdata, ydata4, 'k--', label = 'sigma=2')
plt.plot(xdata,expon(xdata,*popt4),'k-', label = 'fit')

plt.ylabel ('Variance of phase of 1000 oscillators with running average (72,4) fitted to exponential function')
plt.xlabel ('time, hours')
#plt.yscale('log')
#plt.xscale('log')

plt.legend()
plt.show()

#########

#########
popt1,pcov1 = curve_fit(expon,xdata[:1100],ydata1[:1100])
popt2,pcov2 = curve_fit(expon,xdata[:500],ydata2[:500])
popt3,pcov3 = curve_fit(expon,xdata[:400],ydata3[:400])
popt4,pcov4 = curve_fit(expon,xdata[:300],ydata4[:300])
#############################################
#############################################
"""

    


"""
# GRAPH 2

# ONE SIGMA; MANY NUMBERS OF OSCILLATORS

NO TWIST
NO COUPLING

###########
MEAN (x-coordinate)
###########

t = np.linspace(0, 500, 5000)


n = 10 # Number of oscillators
state0 = [2,2]*n
x1 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)))

n = 100 # Number of oscillators
state0 = [2,2]*n
x2 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)))

n = 1000 # Number of oscillators
state0 = [2,2]*n
x3 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n)))

x1x = sep(x1)[0]
x2x = sep(x2)[0]
x3x = sep(x3)[0]


plt.figure(figsize=(20,8))

plt.plot (t, np.mean(x1x,axis=0), label = '10 oscillators')
plt.plot (t, np.mean(x2x,axis=0), label = '100 oscillators')
plt.plot (t, np.mean(x3x,axis=0), label = '1000 oscillators')

plt.ylabel ('Mean of different number of oscillators under 1 sigma variance of omega')
plt.xlabel ('time, hours')


plt.legend()
plt.show()



###########
VAR (x-coordinate)
###########


plt.figure(figsize=(20,8))

plt.plot (t, np.var(x1x,axis=0), label = '10 oscillators')
plt.plot (t, np.var(x2x,axis=0), label = '100 oscillators')
plt.plot (t, np.var(x3x,axis=0), label = '1000 oscillators')

plt.ylabel ('Variance of x-coordinate of different number of oscillators under 1 sigma variance of omega')
plt.xlabel ('time, hours')


plt.legend()
plt.show()


###########
VAR (phase)
###########


plt.figure(figsize=(20,8))

plt.plot (t, phvar(x1)[0], label = '10 oscillators')
plt.plot (t, phvar(x2)[0], label = '100 oscillators')
plt.plot (t, phvar(x3)[0], label = '1000 oscillators')

plt.ylabel ('Variance of phase of different number of oscillators under 1 sigma variance of omega')
plt.xlabel ('time, hours')


plt.legend()
plt.show()
"""
    



"""
###################################################################################################################
#################################################################################################################
#################################################################################################################

NOISY SYSTEM

############################
#################################
######################################################

x1=[[],[]]
x2=[[],[]]
x3=[[],[]]
x4=[[],[]]

x1[1] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/1000 oscillators with E 0.05 only solutions no time.npy")
x2[1] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/1000 oscillators with E 0.1 only solutions no time.npy")
x3[1] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/1000 oscillators with E 0.2 only solutions no time.npy")
x4[1] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/1000 oscillators with E 0.5 only solutions no time.npy")

x1[0] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/1000 oscillators with E 0.05 time.npy")
x2[0] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/1000 oscillators with E 0.1 time.npy")
x3[0] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/1000 oscillators with E 0.2 time.npy")
x4[0] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/1000 oscillators with E 0.5 time.npy")

x1x = sep(x1[1])[0]
x2x = sep(x2[1])[0]
x3x = sep(x3[1])[0]
x4x = sep(x4[1])[0]

#############
#############
#############


n=2 
params = ([0.1]*n,[1]*n,[(np.pi*2)/24]*n,[0.0]*n,[0.0]*n) # alpha (amplitude relaxation rate), A (amplitude), omega, twist, K (coupling), E (white noise, if any)

s1 = ode_rand2(n, 740, (0,0.5,10), [2,2,2,2], params, 0.05)
#s2 = ode_rand2(n, 740, (0,0.5,10), [2,2,2,2], params, 0.05)
#s3 = ode_rand2(n, 740, (0,0.5,10), [2,2,2,2], params, 0.05)
#s4 = ode_rand2(n, 740, (0,0.5,10), [2,2,2,2], params, 0.05)

t1 = s1[0]
x1 = s1[1]

plt.figure(figsize=(20,8))
plt.plot(t1, x1[:,0], label = 'x-coordinate, 1st oscillator')
plt.plot(t1, x1[:,2], label = 'x-coordinate, 2nd oscillator')
plt.legend()
plt.show()
"""


"""
TEST IF TWO SYSTEMS (noisy and non-noisy) GIVE THE SAME OUTPUT IF E=0
n=2 
params = ([0.1]*n,[1]*n,[(np.pi*2)/24]*n,[0.0]*n,[0.0]*n) 
# alpha (amplitude relaxation rate), A (amplitude), omega, twist, K (coupling), E (white noise, if any)
t = np.linspace(0,100,20*100)
state0=[2,2,3,3]

s0 = ode_rand3(n,t,state0,params,0)
t0 = s0[0]
x0 = s0[1]
t1 = np.linspace(0,100,100*20)
x1 = odeint(oscillator_system, state0, np.linspace(0,100,20*100), args = (([0.1]*n,[1]*n,[(np.pi*2)/(24)]*n,[0.0]*n,[0.0]*n, [0]*n)))

plt.plot(t0,x0[:,0], 'o', label = 'ode_rand3')
plt.plot(t1,x1[:,0], '+', label = 'odeint()')

plt.legend()
"""

"""
n=1 
params = ([0.1]*n,[1]*n,[(np.pi*2)/24]*n,[0.0]*n,[0.0]*n)
state0 = [2,2]
x2 = ode_rand3(n,np.linspace(0,200,200*20),state0,params,0.1)
x3 = ode_rand3(n,np.linspace(0,200,200*20),state0,params,0.2)
x4 = ode_rand3(n,np.linspace(0,200,200*20),state0,params,0.3)

plt.figure(figsize=(16,8))
plt.plot(x2[0],x2[1][:,0], label='E=0.1')
plt.plot(x3[0],x3[1][:,0], label='E=0.2')
plt.plot(x4[0],x4[1][:,0], label='E=0.3')
plt.legend()

Smoothened
runx=[]
for i in [x2,x3,x4]:
    runx.append(run_mean(i[1][:,0],72,2))

plt.figure(figsize=(16,8))
plt.plot(runx[0], label='E=0.1')
plt.plot(runx[1], label='E=0.2')
plt.plot(runx[2], label='E=0.3')
plt.legend()
"""

"""


#########################################################
#########################################################

STARTING TO WORK ON THE 1ST PIC

GRAPH 1

n=1000
params = ([0.1]*n,[1]*n,[(np.pi*2)/24]*n,[0.0]*n,[0.0]*n)
state0 = [2,2]*n
t = np.linspace(0,600,600*20)

x1 = ode_rand3(n,t,state0,params,0.05)
x2 = ode_rand3(n,t,state0,params,0.1)
x3 = ode_rand3(n,t,state0,params,0.2)
x4 = ode_rand3(n,t,state0,params,0.5)

x1x = sep(x1[1])[0]
x2x = sep(x2[1])[0]
x3x = sep(x3[1])[0]
x4x = sep(x4[1])[0]

############## Inside one variable
plt.figure(figsize=(16,8))
for i in range(len(x2x)):
    plt.plot(t[0:x2[1].shape[0]], x2x[i], label='x-coord. of osc # ' + str(i))
plt.xlabel('time, hours')
plt.ylabel('X-coordinate of 10 oscillators with E=0.1')
plt.legend()

###########################
    MEAN
##################
plt.figure(figsize=(16,8))

plt.plot(t[0:x1[1].shape[0]], np.mean(x1x, axis=0), label='E=0.05')
plt.plot(t[0:x2[1].shape[0]], np.mean(x2x, axis=0), label='E=0.1')
plt.plot(t[0:x3[1].shape[0]], np.mean(x3x, axis=0), label='E=0.2')
plt.plot(t[0:x4[1].shape[0]], np.mean(x4x, axis=0), label='E=0.5')

plt.xlabel('time, hours')
plt.ylabel('Mean of 1000 oscillators with different noise intensities')
plt.legend()

######## only maxima
plt.figure(figsize=(16,8))

plt.plot(me2(x1x)[0], me2(x1x)[1], 'o-', label='E=0.05')
plt.plot(me2(x2x)[0], me2(x2x)[1], 'o-', label='E=0.1')
plt.plot(me2(x3x)[0], me2(x3x)[1], 'o-', label='E=0.2')
plt.plot(me2(x4x)[0], me2(x4x)[1], 'o-', label='E=0.5')

plt.xlabel('time, hours')
plt.ylabel('Mean (maxima only) of 1000 oscillators with different noise intensities')
plt.legend()

######### using env()
plt.figure(figsize=(16,8))

plt.plot(t[5:x1[1].shape[0]-19], env(np.mean(x1x, axis=0))[5:-19], label='E=0.05')
plt.plot(t[5:x2[1].shape[0]-19], env(np.mean(x2x, axis=0))[5:-19], label='E=0.1')
plt.plot(t[5:x3[1].shape[0]-19], env(np.mean(x3x, axis=0))[5:-19], label='E=0.2')
plt.plot(t[5:x4[1].shape[0]-19], env(np.mean(x4x, axis=0))[5:-19], label='E=0.5')

plt.xlabel('time, hours')
plt.ylabel('Envelope of the mean of 1000 oscillators with different noise intensities')
plt.legend()

#####

##### Using some explicit shit - doesn't work that well as it slightly shifts the curve 
plt.figure(figsize=(16,8))

plt.plot(t[0:10743], run_mean(np.mean(x4x, axis=0), 30, 1), label='E=0.5 smoothened')
plt.plot(t[0:x4[1].shape[0]], np.mean(x4x, axis=0), label='E=0.5 not smoothened')

plt.xlabel('time, hours')
plt.ylabel('Mean of 1000 oscillators with different noise intensities')
#plt.xlim(0,150)
plt.legend()


##### Only maxima after smoothing ! (HERE IS THE RIGHT EXPRESSION FOR THE MOST NOISY SIGNALS)
plt.figure(figsize=(16,8))
m = me(run_mean(np.mean(x4x, axis=0), 30, 1))
m1 = me(np.mean(x4x, axis=0))
plt.plot(m[0], m[1],'+', label='E=0.5 smoothened (30,2), maxima') # This one performs better than anything else
#plt.plot(m1[0], m1[1],'o', label='E=0.5 not smoothened, maxima')
plt.plot(t[0:x4[1].shape[0]], np.mean(x4x, axis=0), label='E=0.5 not smoothened, original data')

plt.xlabel('time, hours')
plt.ylabel('Mean of 1000 oscillators with different noise intensities')
#plt.xlim(0,150)
plt.legend()


##### FINAL MAXIMA

plt.figure(figsize=(16,8))

m4good = me(run_mean(np.mean(x4x, axis=0), 30, 1)) # Smoothened x4 data - E=0.5
m1 = me(np.mean(x1x, axis=0))
m2 = me(np.mean(x2x, axis=0))
m3 = me(np.mean(x3x, axis=0))
m4 = me(np.mean(x4x, axis=0))


plt.plot(m1[0], m1[1],'o-', label = 'E=0.05')
plt.plot(m2[0], m2[1],'o-', label = 'E=0.1')
plt.plot(m3[0], m3[1],'o-', label = 'E=0.2')
#plt.plot(m4[0], m4[1],'+', label = 'E=0.5 raw')
plt.plot(m4good[0], m4good[1],'o-', label='E=0.5 smoothened (30,2), maxima') # This one performs better than anything else

plt.xlabel('time, hours')
plt.ylabel('Mean of 1000 oscillators with different noise intensities - maxima only')
#plt.xlim(0,150)
plt.legend()



####### Final-final MAXIMA (not the best one, though, very strange results)

plt.figure(figsize=(16,8))
e1 = extr(np.mean(x61x, axis=0))
e2 = extr(np.mean(x62x, axis=0))
e3 = extr(np.mean(x63x, axis=0))
e4 = extr(np.mean(x64x, axis=0))
m1 = maxs2(e1)
m2 = maxs2(e2)
m3 = maxs2(e3)
m4 = maxs2(e4)
plt.plot(x61[0],np.mean(x61x, axis=0), label='mean')
plt.plot(x62[0],np.mean(x62x, axis=0), label='mean')
plt.plot(x63[0],np.mean(x63x, axis=0), label='mean')
plt.plot(x64[0],np.mean(x64x, axis=0), label='mean')
#plt.plot(e1[0],e1[1], 'o',label='extrema')
#plt.plot(m1[0],m1[1],'+',label='maxima')
for i in [e1,e2,e3,e4]:
    plt.plot(i[0],i[1],'o', label='extrema')
for i in [m1,m2,m3,m4]:
    plt.plot(i[0],i[1],'+', label = 'maxima')


plt.xlabel('time, hours')
plt.ylabel('Mean of 1000 oscillators with different noise intensities - maxima only')
#plt.xlim(0,150)
plt.title('Mean(x), extrema and maxima')
plt.legend()

####################
FITTING
##########

LIN (x)
###########

m4good = me(run_mean(np.mean(x4x, axis=0), 30, 1)) # Smoothened x4 data - E=0.5
m1 = me(np.mean(x1x, axis=0))
m2 = me(np.mean(x2x, axis=0))
m3 = me(np.mean(x3x, axis=0))
m4 = me(np.mean(x4x, axis=0))

xdata1 = np.array(m1[0])
ydata1 = np.array(m1[1])

xdata2 = np.array(m2[0])
ydata2 = np.array(m2[1])

xdata3 = np.array(m3[0])
ydata3 = np.array(m3[1])

xdata4 = np.array(m4[0])
ydata4 = np.array(m4[1])
    
#xydata = [xdata1,ydata1,xdata2,ydata2,xdata3,ydata3,xdata4,ydata4]  
  

popt1,pcov1 = curve_fit(lin,xdata1,ydata1)
popt2,pcov2 = curve_fit(lin,xdata2[0:5],ydata2[0:5])
popt3,pcov3 = curve_fit(lin,xdata3[0:4],ydata3[0:4])
popt4,pcov4 = curve_fit(lin,xdata4[0:3],ydata4[0:3])

tangents = [popt1[0],popt2[0],popt3[0], popt4[0]]
nt = norm(tangents)
rt = roundl(tangents,4)
angles = slp2ang(tangents)


plt.figure(figsize=(16,8))

plt.plot(xdata1,ydata1, 'ro', label = 's=0.5 maxima')
plt.plot(xdata1, lin(xdata1, *popt1), 'r--', label = 's=0.5 fit')

plt.plot(xdata2,ydata2, 'mo', label = 's=1.0 maxima')
plt.plot(xdata2, lin(xdata2, *popt2), 'm--', label = 's=1.0 fit')

plt.plot(xdata3,ydata3, 'bo', label = 's=1.5 maxima')
plt.plot(xdata3, lin(xdata3, *popt3),'b--', label = 's=1.5 fit')

plt.plot(xdata4,ydata4, 'ko', label = 's=2.0 maxima')
plt.plot(xdata4, lin(xdata4, *popt4), 'k--', label = 's=2.0 fit')

plt.ylabel ('Maxima of mean(x-coordinate) of 1000 oscillators fitted to line')
plt.xlabel ('time, hours')

texttang = ''
for i in rt:
    texttang = texttang+str(i)+' : '
texttang = texttang[:-3]
#plt.text(1,1,'The slopes of curves are\n' + texttang)


plt.ylim(-1.5,1.5)
plt.xlim(10,160)
plt.legend()






#######################
    VAR (X-COORD)
######################
plt.figure(figsize=(16,8))

plt.plot(t[0:x1[1].shape[0]], np.var(x1x, axis=0), label='E=0.05')
plt.plot(t[0:x2[1].shape[0]], np.var(x2x, axis=0), label='E=0.1')
plt.plot(t[0:x3[1].shape[0]], np.var(x3x, axis=0), label='E=0.2')
plt.plot(t[0:x4[1].shape[0]], np.var(x4x, axis=0), label='E=0.5')

plt.xlabel('time, hours')
plt.ylabel('Variance of x-coordinate of 1000 oscillators with different noise intensities')
plt.legend()

#######
running_mean(var(x-coord))
#######
plt.figure(figsize=(16,8))

plt.plot (t[:10562], run_mean(np.var(x1x,axis=0),240), label = 'E=0.05')
#plt.plot (t[:10323], run_mean(np.var(x1x,axis=0),240,1), label = 'E=0.05')
#plt.plot (t[:9364], run_mean(np.var(x1x,axis=0),240,2), label = 'E=0.05')
plt.plot (t[:10562], run_mean(np.var(x2x,axis=0),240), label = 'E=0.1')
plt.plot (t[:10562], run_mean(np.var(x3x,axis=0),240), label = 'E=0.2')
plt.plot (t[:10562], run_mean(np.var(x4x,axis=0),240), label = 'E=0.5')

plt.ylabel ('Variance of x-coordinate of 1000 oscillators with running average (240,1) with different noise intensities')
plt.xlabel ('time, hours')

plt.legend()
plt.show()

###### Fitted to quadratic

xdata=t[:10562]
ydata1=run_mean(np.var(x1x,axis=0),240)
ydata2=run_mean(np.var(x2x,axis=0),240)
ydata3=run_mean(np.var(x3x,axis=0),240)
ydata4=run_mean(np.var(x4x,axis=0),240)

popt1,pcov1 = curve_fit(quad,xdata[:7000],ydata1[:7000])
popt2,pcov2 = curve_fit(quad,xdata[:5000],ydata2[:5000])
popt3,pcov3 = curve_fit(quad,xdata[:3000],ydata3[:3000])
popt4,pcov4 = curve_fit(quad,xdata[:1000],ydata4[:1000])

plt.figure(figsize=(16,8))

plt.plot (xdata, ydata1, 'r--',label = 'E=0.05 data')
plt.plot(xdata,quad(xdata,*popt1),'r-', label = 'fit')

plt.plot (xdata, ydata2,'m--', label = 'E=0.1')
plt.plot(xdata,quad(xdata,*popt2),'m-', label = 'fit')

plt.plot (xdata, ydata3,'b--', label = 'E=0.2')
plt.plot(xdata,quad(xdata,*popt3),'b-', label = 'fit')

plt.plot (xdata, ydata4, 'k--', label = 'E=0.5')
plt.plot(xdata,quad(xdata,*popt4),'k-', label = 'fit')

plt.ylabel ('Variance of x-coordinate of 1000 oscillators with running average (12h,1) fitted to quadratic function')
plt.xlabel ('time, hours')
plt.ylim(-0.1,0.8)
plt.xlim(-5,150)

plt.legend()
plt.show()


######## Fitted to exponential

popt1,pcov1 = curve_fit(expon,xdata[:7000],ydata1[:7000])
popt2,pcov2 = curve_fit(expon,xdata[:5000],ydata2[:5000])
popt3,pcov3 = curve_fit(expon,xdata[:3000],ydata3[:3000])
popt4,pcov4 = curve_fit(expon,xdata[:1000],ydata4[:1000])

########################
    VAR (PHASE)
##################

plt.figure(figsize=(16,8))

plt.plot(t[0:x1[1].shape[0]], phvar(x1[1])[0], label='E=0.05')
plt.plot(t[0:x2[1].shape[0]], phvar(x2[1])[0], label='E=0.1')
plt.plot(t[0:x3[1].shape[0]], phvar(x3[1])[0], label='E=0.2')
plt.plot(t[0:x4[1].shape[0]], phvar(x4[1])[0], label='E=0.5')

plt.xlabel('time, hours')
plt.ylabel('Variance of phase of 1000 oscillators with different noise intensities')
plt.legend()


#######
running_mean(var(phase))
#######

plt.figure(figsize=(16,8))

plt.plot(t[:10562], run_mean(phvar(x1[1])[0], 240), label='E=0.05')
plt.plot(t[:10562], run_mean(phvar(x2[1])[0], 240), label='E=0.1')
plt.plot(t[:10562], run_mean(phvar(x3[1])[0], 240), label='E=0.2')
plt.plot(t[:10562], run_mean(phvar(x4[1])[0], 240), label='E=0.5')

plt.xlabel('time, hours')
plt.ylabel('Variance of phase of 1000 oscillators with different noise intensities with running average (240,1)')
plt.legend()



"""
















"""
#
Power spectra
#
ps = np.abs(np.fft.fft(x4x[0]))**2 # Power spectra computing
time_step = 1/20 # Inverse of the maximal possible frequency (sampling rate, in case of noisy system - 20 times/1.0, so 20Hz)
freqs = np.fft.fftfreq(x4x[0].size, time_step)
idx = np.argsort(freqs)
plt.figure(figsize=(20,8))
plt.plot(freqs[idx], ps[idx])
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('Power spectrum density')
plt.legend()

"""


"""
Picture creation
n001=1
params001 = ([0.1],[1], [(np.pi*2)/24], [0.0], [0.0], [0.0])
t001 = np.linspace(0,50,500)
x001 = odeint(oscillator_system, [1,0] , t001, args=(params001))

fig, (ax1, ax2) = plt.subplots(2, sharey=True, figsize=(14, 10))
ax1.grid()
ax1.plot(t001,x001[:,0], label='')
ax1.plot(t001[359], x001[:,0][359], 'o', markersize=15)
ax1.set(title='Time series for 2-dimentional oscillator', ylabel='x-coordinate (t)')

ax2.plot(t001,x001[:,1], label='')
ax2.plot(t001[359], x001[:,1][359], 'o', markersize=15)
ax2.set(xlabel='time, hours',ylabel='y-coordinate (t)')
ax2.grid()
plt.show()

###
plt.figure(figsize = (7,7))
plt.plot(x001[:,0],x001[:,1], linewidth=1.5, label='limit cycle')
plt.plot(x001[:,0][269],x001[:,1][269],'ro', label='state at 26.9h')
plt.plot([0,x001[:,0][269]],[0,x001[:,1][269]], '-', linewidth=0.7, color="r", label='radius')
plt.plot([-3,3],[0,0],'k-')
plt.plot([0,0],[-3,3],'k-')
plt.ylabel('y-coordinate (t)')
plt.xlabel('x-coordinate (t)')
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.title('Phase plane plot')
plt.text(0.6,0.85, '(0.71,0.69)\n 26.9h from (1,0)')
plt.grid()
plt.legend()
plt.show()
#########

Phase variance is oscillating:

    n=2
t = np.linspace(0, 1000, 1000)
state0 = [1,0]*n
x01 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(25.5), (np.pi*2)/(22.5)],[0.0]*n,[0.0]*n, [0.0]*n)))

plt.plot (t, phvar(x01)[0], label = 'sigma=0.5')
plt.xlim(0,400)
##########

Desync in heterogeneity
#######
n=10
t = np.linspace(0, 1000, 1000)
state0 = [1,0]*n
x03 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n, [0.0]*n)))
x03x = sep(x03)[0]
plt.figure (figsize=(14,10))
for i in range(10):
    plt.plot(t,x03x[i])
plt.xlim(0,100)
#plt.grid()
plt.ylabel('x-coordinate')
plt.xlabel('time, hours')
plt.title('Desync in heterogenous system, N=10', fontsize=26)
##########

Noise example

n=1
t9 = np.linspace(0,100,100*20)
state09 = [1,0]*n
params9 = ([0.1]*n,[1]*n,[(np.pi*2)/24]*n,[0.0]*n,[0.0]*n)
x91 = ode_rand3(n,t9,state09,params9,0.05)
x92 = ode_rand3(n,t9,state09,params9,0.1)
x93 = ode_rand3(n,t9,state09,params9,0.2)
x94 = ode_rand3(n,t9,state09,params9,0.5)

x91x = sep(x91[1])[0]
x92x = sep(x92[1])[0]
x93x = sep(x93[1])[0]
x94x = sep(x94[1])[0]

plt.figure(figsize=(14,10))
plt.plot(t9[:1801],x91x[0], label='E=0.05')
plt.plot(t9[:1801],x92x[0], label='E=0.1')
plt.plot(t9[:1801],x93x[0], label='E=0.2')
#plt.plot(t9[:1801],x94x[0], label='E=0.5')
plt.legend()
plt.title('Examples of noisy oscillations', fontsize=26)
plt.ylabel('x-coordinate')
plt.xlabel('time, hours')
"""

"""
What if initial conditions matter?

n=1000
t5 = np.linspace(0,400,4000)
state05 = [1,0]*n
x51 = odeint(oscillator_system, state05, t5, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 0.5*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n, [0.0]*n)))
x52 = odeint(oscillator_system, state05, t5, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n, [0.0]*n)))
x53 = odeint(oscillator_system, state05, t5, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1.5*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n, [0.0]*n)))
x54 = odeint(oscillator_system, state05, t5, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 2*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n, [0.0]*n)))

x51x = sep(x51)[0]
x52x = sep(x52)[0]
x53x = sep(x53)[0]
x54x = sep(x54)[0]


# Same but noise
n=1000
t6 = np.linspace(0,600,600*20)
state06 = [1,0]*n
params6 = ([0.1]*n,[1]*n,[(np.pi*2)/24]*n,[0.0]*n,[0.0]*n)
x61 = ode_rand3(n,t6,state06,params6,0.05)
x62 = ode_rand3(n,t6,state06,params6,0.1)
x63 = ode_rand3(n,t6,state06,params6,0.2)
x64 = ode_rand3(n,t6,state06,params6,0.5)

x61x = sep(x61[1])[0]
x62x = sep(x62[1])[0]
x63x = sep(x63[1])[0]
x64x = sep(x64[1])[0]

######################################################################################################
########################################################################################################

#Kind of animation (not really)

for i in range(100):
    plt.figure(figsize=(8,8))
    plt.plot(x1x,x1y, 'b-', label='limit cycle')
    plt.plot(x1x[i],x1y[i], 'ro')
    plt.xlabel('x-coord')
    plt.ylabel('y-coord')
    plt.show()

"""