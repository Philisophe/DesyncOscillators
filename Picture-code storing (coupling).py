#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 17:21:12 2018

@author: kalashnikov
"""
"""
n = 1000 # Number of oscillators
t = np.linspace(0, 700, 7000)
state0 = [1,0]*n

x1 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.05]*n, [0.0]*n)))
x2 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.1]*n, [0.0]*n)))
x3 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.15]*n, [0.0]*n)))
x4 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.2]*n, [0.0]*n)))
x5 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.3]*n, [0.0]*n)))
x6 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.5]*n, [0.0]*n)))

###
t = np.linspace(0, 700, 7000)
n = 1000
state0 = [1,0]*n
x7 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.01]*n, [0.0]*n)))
x8 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.02]*n, [0.0]*n)))
x9 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.04]*n, [0.0]*n)))
x10 = odeint(oscillator_system, state0, t, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.07]*n, [0.0]*n)))
###


x1x = sep(x1)[0]
x2x = sep(x2)[0]
x3x = sep(x3)[0]
x4x = sep(x4)[0]
x5x = sep(x5)[0]
x6x = sep(x6)[0]

per1  = pers(x1x,300)
per2  = pers(x2x,300)
per3  = pers(x3x,300)
per4  = pers(x4x,300)

per7  = pers(x7x,300)
per8  = pers(x8x,300)
per9  = pers(x9x,300)
per10  = pers(x10x,300)

# SAVED
np.save("/home/kalashnikov/Code/Variables for my code/Coupling/Heterogeneity/1000 coupled oscillators without noise s=1 and K=0.05",x1)
np.save("/home/kalashnikov/Code/Variables for my code/Coupling/Heterogeneity/1000 coupled oscillators without noise s=1 and K=0.1",x2)
np.save("/home/kalashnikov/Code/Variables for my code/Coupling/Heterogeneity/1000 coupled oscillators without noise s=1 and K=0.15",x3)
np.save("/home/kalashnikov/Code/Variables for my code/Coupling/Heterogeneity/1000 coupled oscillators without noise s=1 and K=0.2",x4)
np.save("/home/kalashnikov/Code/Variables for my code/Coupling/Heterogeneity/1000 coupled oscillators without noise s=1 and K=0.3",x5)
np.save("/home/kalashnikov/Code/Variables for my code/Coupling/Heterogeneity/1000 coupled oscillators without noise s=1 and K=0.5",x6)
np.save("/home/kalashnikov/Code/Variables for my code/Coupling/Heterogeneity/1000 coupled oscillators without noise time",t)


BUT AFTER THAT I USED ONLY THE IMPORTANT DATASETS:

x4 = np.load("/home/kalashnikov/Code/Variables for my code/Coupling/Heterogeneity/1000 coupled oscillators without noise s=1 and K=0.5.npy")
x3 = np.load("/home/kalashnikov/Code/Variables for my code/Coupling/Heterogeneity/1000 coupled oscillators without noise s=1 and K=0.2.npy")
x2 = np.load("/home/kalashnikov/Code/Variables for my code/Coupling/Heterogeneity/1000 coupled oscillators without noise s=1 and K=0.1.npy")
x1 = np.load("/home/kalashnikov/Code/Variables for my code/Coupling/Heterogeneity/1000 coupled oscillators without noise s=1 and K=0.05.npy")



######
# Checking where is the asymptote for some data
t2 = np.linspace(0,2000,10000)
x5 = odeint(oscillator_system, state0, t2, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.02]*n, [0.0]*n)))
x6 = odeint(oscillator_system, state0, t2, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 1*i) for i in np.random.randn(n)],[0.0]*n,[0.03]*n, [0.0]*n)))
x5x = sep(x5)[0]
x6x = sep(x6)[0]

# Now some noisy shit
params3 = ([0.1]*n,[1]*n,[(np.pi*2)/24]*n,[0.0]*n,[0.0]*n)
t3 = np.linspace(0,700,700*20)
x7 = ode_rand3(n,t3,state0,params3,0.05)
x8 = ode_rand3(n,t3,state0,params3,0.1)
x9 = ode_rand3(n,t3,state0,params3,0.2)
x7x = sep(x7[1])[0]
x8x = sep(x8[1])[0]
x9x = sep(x9[1])[0]
######


"""


"""

plt.rc('font', size=16)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=22)  # fontsize of the figure title

###
RAW data
###
plt.figure(figsize=(30,8))
for i in range(1000):
    plt.plot(t, x1x[i])
plt.ylabel ('x-coordinate')
plt.xlabel ('time, hours')
#plt.ylim(-1.5,2.5)
plt.xlim(100,300)
plt.title('x-coordinate of 1000 coupled oscillators with K=0.05', fontsize=22)
plt.legend()
plt.show()



#MEAN VS TIME
m1 = np.mean(x1x,axis=0)
m2 = np.mean(x2x,axis=0)
m3 = np.mean(x3x,axis=0)
m4 = np.mean(x4x,axis=0)

plt.figure(figsize=(14,10))

plt.plot (t, np.mean(x1x,axis=0),'k-', label = 's=1, K=0.02')
plt.plot (t, np.mean(x2x,axis=0),'b-', label = 's=1, K=0.03')
plt.plot (t, np.mean(x3x,axis=0),'m-', label = 's=1, K=0.04')
plt.plot (t, np.mean(x4x,axis=0),'r-', label = 's=1, K=0.05')

plt.ylabel ('x-coordinate')
plt.xlabel ('time, hours')
#plt.ylim(-1.5,2.5)
#plt.xlim(-10,400)
plt.title('Mean (x-coordinate) of coupled heterogenous oscillators', fontsize=22)
plt.legend()
plt.show()
######
Maxima and mean vs. time
###
plt.figure(figsize=(14,10))
plt.plot (t[maxs4(m1)], m1[maxs4(m1)],'ko', label = 's=1, K=0.05 maxima')
plt.plot(t,m1,'k-', label = 's=1, K=0.05 data')
plt.plot (t[maxs4(m2)], m2[maxs4(m2)],'bo', label = 's=1, K=0.1 maxima')
plt.plot(t,m2,'b-', label = 's=1, K=0.1 data')
plt.plot (t[maxs4(m3)], m3[maxs4(m3)],'mo', label = 's=1, K=0.2 maxima')
plt.plot(t,m3,'m-', label = 's=1, K=0.2 data')
plt.plot (t[maxs4(m4)], m4[maxs4(m4)],'ro', label = 's=1, K=0.5 maxima')
plt.plot(t, m4,'r-', label = 's=1, K=0.5 data')

plt.ylabel ('x-coordinate')
plt.xlabel ('time, hours')
#plt.ylim(-1.5,2.5)
#plt.xlim(-10,400)
plt.title('Mean + maxima (x-coordinate) of coupled heterogenous oscillators', fontsize=22)
plt.legend()
plt.grid()
plt.show()

##########
Maxima only (mean(x)) + IC
#######

plt.figure(figsize=(14,10))
plt.plot (np.insert(t[maxs4(m1)],0,0), np.insert(m1[maxs4(m1)],0,1),'ko-', label = 's=1, K=0.02 maxima')
#plt.plot(t,m1,'k-', label = 's=1, K=0.02 data')
plt.plot (np.insert(t[maxs4(m2)],0,0), np.insert(m2[maxs4(m2)],0,1),'bo-', label = 's=1, K=0.03 maxima')
#plt.plot(t,m2,'b-', label = 's=1, K=0.03 data')
plt.plot (np.insert(t[maxs4(m3)],0,0), np.insert(m3[maxs4(m3)],0,1),'mo-', label = 's=1, K=0.04 maxima')
#plt.plot(t,m3,'m-', label = 's=1, K=0.04 data')
plt.plot (np.insert(t[maxs4(m4)],0,0), np.insert(m4[maxs4(m4)],0,1),'ro-', label = 's=1, K=0.05 maxima')
#plt.plot(t, m4,'r-', label = 's=1, K=0.05 data')
plt.ylabel ('x-coordinate')
plt.xlabel ('time, hours')
#plt.ylim(-1.5,2.5)
#plt.xlim(-10,400)
plt.title('Maxima of mean (x-coordinate) of coupled heterogenous oscillators', fontsize=22)
plt.legend()
plt.grid()
plt.show()

#######################################
# Var(x) vs. time
################
v1=np.var(x1x,axis=0)
v2=np.var(x2x,axis=0)
v3=np.var(x3x,axis=0)
v4=np.var(x4x,axis=0)
v5=np.var(x5x,axis=0)
v6=np.var(x6x,axis=0)

##
plt.figure(figsize=(14,10))

plt.plot (t, np.var(x1x,axis=0), label = 's=1, K=0.02')
plt.plot (t, np.var(x2x,axis=0), label = 's=1, K=0.03')
plt.plot (t, np.var(x3x,axis=0), label = 's=1, K=0.04')
plt.plot (t, np.var(x4x,axis=0), label = 's=1, K=0.05')

plt.ylabel ('variance of x')
plt.xlabel ('time, hours')
#plt.xlim(0,200)
plt.title ('Variance of x-coordinate of heterogenous coupled oscillators (raw)', fontsize=22)
plt.legend()
plt.grid()
plt.show()

######
MAXIMA only
#
plt.figure(figsize=(14,10))

plt.plot (t[maxs4(np.var(x1x,axis=0))], np.var(x1x,axis=0)[maxs4(np.var(x1x,axis=0))],'k-', label = 's=1, K=0.05') # Alternatively: plt.plot (t[maxs4(v1)], v1[maxs4(v1)],'k-', label = 's=1, K=0.05')
plt.plot (t[maxs4(np.var(x2x,axis=0))], np.var(x2x,axis=0)[maxs4(np.var(x2x,axis=0))],'b-', label = 's=1, K=0.1')
plt.plot (t[maxs4(np.var(x3x,axis=0))], np.var(x3x,axis=0)[maxs4(np.var(x3x,axis=0))],'m-', label = 's=1, K=0.2')
plt.plot (t[maxs4(np.var(x4x,axis=0))], np.var(x4x,axis=0)[maxs4(np.var(x4x,axis=0))],'r-', label = 's=1, K=0.5')

plt.ylabel ('x-coordinate')
plt.xlabel ('time, hours')
#plt.ylim(-1.5,2.5)
#plt.xlim(-10,400)
plt.title('Var (x-coordinate) of 1000 heterogenous oscillators', fontsize=22)
plt.legend()
plt.show()

# Another option
plt.plot (t[maxs4(v1)], v1[maxs4(v1)],'k-', label = 's=1, K=0.05')
plt.plot (t[maxs4(v2)], v2[maxs4(v2)],'b-', label = 's=1, K=0.1')
plt.plot (t[maxs4(v3)], v3[maxs4(v3)],'m-', label = 's=1, K=0.2')
plt.plot (t[maxs4(v4)], v4[maxs4(v4)],'r-', label = 's=1, K=0.5')

# Maxima + starting values
plt.figure(figsize=(14,10))
plt.plot (np.insert(t[maxs4(v1)],0,0), np.insert(v1[maxs4(v1)],0,0),'k-', label = 's=1, K=0.02')
plt.plot (np.insert(t[maxs4(v2)],0,0), np.insert(v2[maxs4(v2)],0,0),'b-', label = 's=1, K=0.03')
plt.plot (np.insert(t[maxs4(v3)],0,0), np.insert(v3[maxs4(v3)],0,0),'m-', label = 's=1, K=0.04')
plt.plot (np.insert(t[maxs4(v4)],0,0), np.insert(v4[maxs4(v4)],0,0),'r-', label = 's=1, K=0.05')
plt.ylabel ('variance (x)')
plt.xlabel ('time, hours')
#plt.ylim(-1.5,2.5)
#plt.xlim(-5,100)
plt.title('Variance of x-coordinate of heterogenous coupled oscillators', fontsize=22)
plt.legend()
plt.grid()
plt.show()


########
VAR (phase) (raw, smooth, maxima)
########
ph1 = phvar(x1)[0]
ph2 = phvar(x2)[0]
ph3 = phvar(x3)[0]
ph4 = phvar(x4)[0]

ph1rm = run_mean(ph1,40)
ph2rm = run_mean(ph2,40)
ph3rm = run_mean(ph3,40)
ph4rm = run_mean(ph4,40)

plt.figure(figsize=(14,10))
plt.plot (t[:6881], run_mean(phvar(x1)[0],120),'k--', label = 'K=0.05 run_mean')
plt.plot (t, phvar(x1)[0],'k-', label = 'K=0.05')
plt.plot (t[maxs4(ph1)], ph1[maxs4(ph1)],'ko', label = 'K=0.05 maxima')

plt.plot (t[:6881], run_mean(phvar(x2)[0],120),'b--', label = 'K=0.1 run_mean')
plt.plot (t, phvar(x2)[0],'b-', label = 'K=0.1')
plt.plot (t[maxs4(ph2)], ph2[maxs4(ph2)],'bo', label = 'K=0.1 maxima')

plt.plot (t[:6881], run_mean(phvar(x3)[0],120),'m--', label = 'K=0.2 run_mean')
plt.plot (t, phvar(x3)[0],'m-', label = 'K=0.2')
plt.plot (t[maxs4(ph3)], ph3[maxs4(ph3)],'mo', label = 'K=0.2 maxima')

plt.plot (t[:6881], run_mean(phvar(x4)[0],120),'r--', label = 'K=0.5 run_mean')
plt.plot (t, phvar(x4)[0],'r-', label = 'K=0.5')
plt.plot (t[maxs4(ph4)], ph4[maxs4(ph4)],'ro', label = 'K=0.5 maxima')

plt.ylabel ('Variance of phase')
plt.xlabel ('time, hours')
plt.title('Variance of phase of coupled heterogenous oscillators (raw, smooth and maxima)', fontsize=26)
#plt.xlim(-5,150)
plt.grid()
plt.legend()
plt.show()

####
Smoothened a little
###
plt.figure(figsize=(14,10))
#plt.plot (t[:6881], run_mean(phvar(x1)[0],120),'k--', label = 'K=0.05 run_mean')
plt.plot (t, phvar(x1)[0],'k-', label = 'K=0.02')
plt.plot (t[maxs4(ph1rm)], ph1rm[maxs4(ph1rm)],'ko', label = 'maxima')

#plt.plot (t[:6881], run_mean(phvar(x2)[0],120),'b--', label = 'K=0.1 run_mean')
plt.plot (t, phvar(x2)[0],'b-', label = 'K=0.03')
plt.plot (t[maxs4(ph2rm)], ph2rm[maxs4(ph2rm)],'bo', label = 'maxima')

#plt.plot (t[:6881], run_mean(phvar(x3)[0],120),'m--', label = 'K=0.2 run_mean')
plt.plot (t, phvar(x3)[0],'m-', label = 'K=0.04')
plt.plot (t[maxs4(ph3rm)], ph3rm[maxs4(ph3rm)],'mo', label = 'maxima')

#plt.plot (t[:6881], run_mean(phvar(x4)[0],120),'r--', label = 'K=0.5 run_mean')
plt.plot (t, phvar(x4)[0],'r-', label = 'K=0.05')
plt.plot (t[maxs4(ph4rm)], ph4rm[maxs4(ph4rm)],'ro', label = 'maxima')

plt.ylabel ('Variance of phase')
plt.xlabel ('time, hours')
plt.title('Variance of phase of coupled heterogenous oscillators (raw and maxima)', fontsize=26)
#plt.xlim(502,512)
plt.grid()
plt.legend()


###################################################################

Coupling
###########################








# Period estimation/calculation
per7  = pers(x7x,300)
per8  = pers(x8x,300)
per9  = pers(x9x,300)
per10  = pers(x10x,300)
plt.hist(per7, label = 'K=0.01')
plt.hist(per8, label = 'K=0.02')
plt.hist(per9, label = 'K=0.04')
plt.legend()
plt.grid()

##

bins1 = np.arange(22,26.5,0.5)
plt.figure(figsize=(10,10))
plt.hist(per4, bins1,label = 'K = 0.07', density=True)
plt.hist(per3, bins1,label = 'K = 0.04', density=True)
plt.hist(per6, bins1, label = 'K = 0.03', density = True)
plt.hist(per2, bins1,label = 'K = 0.02', density=True)
plt.hist(per1, bins1, label = 'K = 0.01', density=True)
plt.legend()
plt.xlim(22,26)
plt.grid()
"""