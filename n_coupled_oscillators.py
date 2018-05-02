# This code integrates over a system of ODEs of coupled oscillators with parameters.
# The single oscillator is implemented as a modified Poincare oscillator (with twist - parameter, linking period and amplitude).


from scipy.integrate import odeint
from scipy.signal import hilbert
from scipy.optimize import curve_fit
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


def extr(x):
    diff = np.diff(np.sign(np.diff(x)))
    extrT=[]
    extrVal=[]
    for i in range(len(diff)):
        if diff[i]!=0:
            extrVal.append(np.mean(x[i:i+2]))
            extrT.append(np.mean(t[i:i+2]))
    return [extrT,extrVal]


def maxs(list_extr):
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

#Running average
# x - data, N - window size, N2 - number of runs (if N2=0 - 1 run, if N2=1 - 2 runs etc.)
def run_mean(x, N,N2=0):
    if N2==0:
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    else:
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return run_mean(((cumsum[N:] - cumsum[:-N]) / float(N)),N,N2-1)


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



# Calculates envelope
def env(x):
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
##########################################################################################################
##########################################################################################################


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
popt3,pcov3 = curve_fit(lin,xdata3[0:5],ydata3[0:5])
popt4,pcov4 = curve_fit(lin,xdata4[0:5],ydata4[0:5])


plt.figure(figsize=(10,6))
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

plt.xlim(20,250)
plt.ylim(-1.7,1.3)
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


###### Fitting to the exponential

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

###################
QUADRATIC

popt1,pcov1 = curve_fit(quad,xdata,ydata1)
popt2,pcov2 = curve_fit(quad,xdata,ydata2)
popt3,pcov3 = curve_fit(quad,xdata,ydata3)
popt4,pcov4 = curve_fit(quad,xdata,ydata4)


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
    

















