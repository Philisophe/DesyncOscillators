#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:47:17 2018

@author: kalashnikov
"""
"""
##############################
# STARTING THE ENGINE
#############################
n=1000
t = np.linspace(0,600,600*20)
state0 = [1,0]*n
params = ([0.1]*n,[1]*n,[(np.pi*2)/24]*n,[0.0]*n,[0.0]*n)


x1=[[],[]]
x2=[[],[]]
x3=[[],[]]
x4=[[],[]]

x1[1] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/Other state0/1000 oscillators with E 0.05 and state0 [1,0].npy")
x2[1] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/Other state0/1000 oscillators with E 0.1 and state0 [1,0].npy")
x3[1] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/Other state0/1000 oscillators with E 0.2 and state0 [1,0].npy")
x4[1] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/Other state0/1000 oscillators with E 0.5 and state0 [1,0].npy")

x1[0] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/Other state0/1000 oscillators with E 0.05 and state0 [1,0] time.npy")
x2[0] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/Other state0/1000 oscillators with E 0.1 and state0 [1,0] time.npy")
x3[0] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/Other state0/1000 oscillators with E 0.2 and state0 [1,0] time.npy")
x4[0] = np.load("/home/kalashnikov/Code/Variables for my code/Noise/Other state0/1000 oscillators with E 0.5 and state0 [1,0] time.npy")

############
FOR PHASE-data
####
n=1000
t6 = np.linspace(0,600,600*20)
state06 = [1,0]*n
params6 = ([0.1]*n,[1]*n,[(np.pi*2)/24]*n,[0.0]*n,[0.0]*n)
x61 = ode_rand3(n,t6,state06,params6,0.05)
x62 = ode_rand3(n,t6,state06,params6,0.1)
x63 = ode_rand3(n,t6,state06,params6,0.2)
x64 = ode_rand3(n,t6,state06,params6,0.5)
############
np.save("/home/kalashnikov/Code/Variables for my code/2nd attempt/1000 oscillators with E 0.05 and state0 [1,0]",x61[1])
np.save("/home/kalashnikov/Code/Variables for my code/1000 oscillators with E 0.05 and state0 [1,0] v2",x61[1])
np.save("/home/kalashnikov/Code/Variables for my code/1000 oscillators with E 0.1 and state0 [1,0] v2",x62[1])
np.save("/home/kalashnikov/Code/Variables for my code/1000 oscillators with E 0.2 and state0 [1,0] v2",x63[1])
np.save("/home/kalashnikov/Code/Variables for my code/1000 oscillators with E 0.5 and state0 [1,0] v2",x64[1])
x1 = x61
x2 = x62
x3 = x63
x4 = x64

###########

#x1x = sep(x1[1])[0]
#x2x = sep(x2[1])[0]
#x3x = sep(x3[1])[0]
#x4x = sep(x4[1])[0]

#x1x = np.array(x1x)
#x2x = np.array(x2x)
#x3x = np.array(x3x)
#x4x = np.array(x4x)


#### To use all the code below (until var(phase)) - 
# first - load the data in files
# Then - x1x = sep()
# Then - x1 = x1x

# Only in var(phase)
# Here - just import the data again as x1,x2,x3,x4

### INSTEAD OF x1x USE x1[1]
### For the 1st oscillator in x1[1] use x1[1][0]


def me4(x):
    return maxs3(extr(x))

"""



"""
################################################## NOISE
##################################################

############## Inside one variable
plt.figure(figsize=(20,8))

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

for i in range(10):
    plt.plot(x2[0], x2[1][i], label='x-coord. of osc #' + str(i))
plt.xlabel('time, hours')
plt.ylabel('x-coordinate')
plt.title('Desynchronization of 10 oscillators with mild noise (E=0.1)', fontsize=16)
plt.xlim(-10,300)
#plt.legend()


####### MEAN vs. TIME

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

plt.figure(figsize=(16,8))

plt.plot(x1[0], np.mean(x1[1], axis=0), 'k-', label='E=0.05')
plt.plot(x2[0], np.mean(x2[1], axis=0), 'b-', label='E=0.1')
plt.plot(x3[0], np.mean(x3[1], axis=0), 'm-', label='E=0.2')
plt.plot(x4[0], np.mean(x4[1], axis=0), 'r-', label='E=0.5')

plt.xlabel('time, hours')
plt.ylabel('x-coordinate')
plt.title('Mean (x-coodrinate) of 1000 noisy oscillators with different noise intensities', fontsize=16)
plt.legend()



######## MEAN + MAX vs. TIME

plt.figure(figsize=(16,8))

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

plt.plot(x1[0], np.mean(x1[1], axis=0), 'k-', label='E=0.05')
plt.plot(x2[0], np.mean(x2[1], axis=0), 'b-', label='E=0.1')
plt.plot(x3[0], np.mean(x3[1], axis=0), 'm-', label='E=0.2')
plt.plot(x4[0], np.mean(x4[1], axis=0), 'r-', label='E=0.5')

plt.plot(me4(np.mean(x1[1],axis=0))[0], me4(np.mean(x1[1],axis=0))[1], 'ko', label="maxima sigma 0.5")
plt.plot(me4(np.mean(x2[1],axis=0))[0], me4(np.mean(x2[1],axis=0))[1], 'bo', label="maxima sigma 1.0")
plt.plot(me4(np.mean(x3[1],axis=0))[0], me4(np.mean(x3[1],axis=0))[1], 'mo', label="maxima sigma 1.5")
plt.plot(me4(np.mean(x4[1],axis=0))[0], me4(np.mean(x4[1],axis=0))[1], 'ro', label="maxima sigma 2.0")

plt.xlabel('time, hours')
plt.ylabel('x-coordinate')
plt.title('Mean (x-coodrinate) of 1000 noisy oscillators with different noise intensities', fontsize=16)
plt.legend()


####### MAXIMA

plt.figure(figsize=(16,8))

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

m4good = me4 (run_mean (np.mean(x4[1], axis=0), 30, 1)) # Smoothened (30,2)
m1 = me4 (run_mean (np.mean(x1[1], axis=0), 30, 0)) # Smoothened (30,1)
m2 = me4 (run_mean (np.mean(x2[1], axis=0), 30, 0))
m3 = me4 (run_mean (np.mean(x3[1], axis=0), 30, 0))
m4 = me4 (run_mean (np.mean(x4[1], axis=0), 30, 0))


plt.plot(m1[0], m1[1],'ko-', label = 'E=0.05')
plt.plot(m2[0], m2[1],'bo-', label = 'E=0.1')
plt.plot(m3[0], m3[1],'mo-', label = 'E=0.2')
#plt.plot(m4[0], m4[1],'+', label = 'E=0.5 raw')
plt.plot(m4good[0], m4good[1],'ro-', label='E=0.5') 

plt.xlabel('time, hours')
plt.ylabel('x-coordinate')
#plt.xlim(0,150)
#plt.ylim(-0.80,1.05)

plt.title('Maxima of means (x-coordinate) of 1000 noisy oscillators with different noise intensities', fontsize=16)
plt.legend(loc=1)




########## FITTING TO THE LINE

plt.figure(figsize=(16,8))
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

m1 = me4 (run_mean (np.mean(x1[1], axis=0), 30, 0)) # Smoothened (30,1)
m2 = me4 (run_mean (np.mean(x2[1], axis=0), 30, 0))
m3 = me4 (run_mean (np.mean(x3[1], axis=0), 30, 0))
m4 = me4 (run_mean (np.mean(x4[1], axis=0), 30, 1)) # Smoothened (30,2)
#m4_BAD = me4 (run_mean (np.mean(x4[1], axis=0), 30, 0))

xdata1 = np.array(m1[0])
ydata1 = np.array(m1[1])
xdata2 = np.array(m2[0])
ydata2 = np.array(m2[1])
xdata3 = np.array(m3[0])
ydata3 = np.array(m3[1])
xdata4 = np.array(m4[0])
ydata4 = np.array(m4[1])

popt1,pcov1 = curve_fit(lin,xdata1,ydata1)
popt2,pcov2 = curve_fit(lin,xdata2[1:10],ydata2[1:10])
popt3,pcov3 = curve_fit(lin,xdata3[0:5],ydata3[0:5])
popt4,pcov4 = curve_fit(lin,xdata4[0:3],ydata4[0:3])

tangents = [popt1[0],popt2[0],popt3[0], popt4[0]]
nt = norm(tangents)
rt = roundl(tangents,4)
angles = slp2ang(tangents)

plt.plot(xdata1,ydata1, 'ko', label = 'E=0.05 maxima')
plt.plot(xdata1, lin(xdata1, *popt1), 'k--', label = 'E=0.05 fit')
plt.plot(xdata2,ydata2, 'bo', label = 'E=0.1 maxima')
plt.plot(xdata2, lin(xdata2, *popt2), 'b--', label = 'E=0.1 fit')
plt.plot(xdata3,ydata3, 'mo', label = 'E=0.2 maxima')
plt.plot(xdata3, lin(xdata3, *popt3),'m--', label = 'E=0.2 fit')
plt.plot(xdata4,ydata4, 'ro', label = 'E=0.5 maxima')
plt.plot(xdata4, lin(xdata4, *popt4), 'r--', label = 'E=0.5 fit')

plt.xlabel('time, hours')
plt.ylabel('x-coordinate')
texttang = ''
for i in rt:
    texttang = texttang+str(i)+' : '
texttang = texttang[:-3]
plt.text(1,1.1,'The slopes:\n' + texttang)

plt.ylim(-0.05,1.2)
#plt.xlim(-5,300)
plt.legend(loc=1)
plt.title('Maxima of means (x-coordinate) of 1000 noisy oscillators with different noise intensities, linear fit', fontsize=16)



####### EXPONENTIAL FIT

plt.figure(figsize=(16,8))
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

m1 = me4 (run_mean (np.mean(x1[1], axis=0), 30, 0)) # Smoothened (30,1)
m2 = me4 (run_mean (np.mean(x2[1], axis=0), 30, 0))
m3 = me4 (run_mean (np.mean(x3[1], axis=0), 30, 0))
m4 = me4 (run_mean (np.mean(x4[1], axis=0), 30, 1)) # Smoothened (30,2)
#m4_BAD = me4 (run_mean (np.mean(x4[1], axis=0), 30, 0))

xdata1 = np.array(m1[0])
ydata1 = np.array(m1[1])
xdata2 = np.array(m2[0])
ydata2 = np.array(m2[1])
xdata3 = np.array(m3[0])
ydata3 = np.array(m3[1])
xdata4 = np.array(m4[0])
ydata4 = np.array(m4[1])

# This shit won't work as predicted
#popt1,pcov1 = curve_fit(expon,xdata1,ydata1, maxfev = 10000)
#popt2,pcov2 = curve_fit(expon,xdata2,ydata2, maxfev = 10000)
#popt3,pcov3 = curve_fit(expon,xdata3,ydata3, maxfev = 10000)
#popt4,pcov4 = curve_fit(expon,xdata4,ydata4, maxfev = 10000)

# To obtain these - use Desmos.com
# Copy-paste datapoints from xdata/ydata to the tables; then regression
popt1 = [0.362215, 0.00099855, 0.637115] # R2 = 0.9982
popt2 = [1.27066, 0.000797632, -0.282885] # R2 = 0.9983
popt3 = [1.04445, 0.0052881, -0.032042] # R2 = 0.9984
popt4 = [1.00651, 0.0325793, 0.00844607] # R2 = 0.9837


plt.plot(xdata1,ydata1, 'ko', label = 'E=0.05 maxima')
plt.plot(xdata1, expon(xdata1, *popt1), 'k--', label = 'E=0.05 fit')
plt.plot(xdata2,ydata2, 'bo', label = 'E=0.1 maxima')
plt.plot(xdata2, expon(xdata2, *popt2), 'b--', label = 'E=0.1 fit')
plt.plot(xdata3,ydata3, 'mo', label = 'E=0.2 maxima')
plt.plot(xdata3, expon(xdata3, *popt3),'m--', label = 'E=0.2 fit')
plt.plot(xdata4,ydata4, 'ro', label = 'E=0.5 maxima')
plt.plot(xdata4, expon(xdata4, *popt4), 'r--', label = 'E=0.5 fit')

plt.xlabel('time, hours')
plt.ylabel('x-coordinate')

plt.legend(loc=1)
plt.title('Maxima of means (x-coordinate) of 1000 noisy oscillators with different noise intensities, exponential fit', fontsize=16)



#########################
#########################
##########################
###########################

##############
################## VAR (x)
################

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

plt.figure(figsize=(16,8))

plt.plot(x1[0], np.var(x1[1], axis=0), 'k-', label='E=0.05')
plt.plot(x2[0], np.var(x2[1], axis=0), 'b-', label='E=0.1')
plt.plot(x3[0], np.var(x3[1], axis=0), 'm-', label='E=0.2')
plt.plot(x4[0], np.var(x4[1], axis=0), 'r-', label='E=0.5')

plt.xlabel('time, hours')
plt.ylabel('x-coordinate variance')
plt.title('Variance (x-coodrinate) of 1000 noisy oscillators with different noise intensities', fontsize=16)
plt.legend()

############
####SMOOOTH
###





plt.figure(figsize=(16,8))

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

plt.plot (x1[0][:10562], run_mean(np.var(x1[1],axis=0),240), 'k-', label = 'E=0.05')
#plt.plot (x1[0][:10323], run_mean(np.var(x1[1],axis=0),240,1),'k-', label = 'E=0.05')
#plt.plot (x1[0][:9364], run_mean(np.var(x1[1],axis=0),240,2),'k-', label = 'E=0.05')
plt.plot (x2[0][:10562], run_mean(np.var(x2[1],axis=0),240), 'b-',label = 'E=0.1')
plt.plot (x3[0][:10562], run_mean(np.var(x3[1],axis=0),240),'m-', label = 'E=0.2')
plt.plot (x4[0][:10562], run_mean(np.var(x4[1],axis=0),240),'r-', label = 'E=0.5')
#plt.plot(x4[0][0:100],np.var(x4[1], axis=0)[0:100], 'r-')

plt.ylabel ('x-coordinate variance')
plt.xlabel ('time, hours')
plt.title('Variance (x-coodrinate) of 1000 noisy oscillators with different noise intensities (smoothened)', fontsize=16)
plt.legend(loc=1)
plt.show()



#######
# FITTING TO QUAD
######

xdata1 = x1[0][:10562]
xdata2 = x2[0][:10562]
xdata3 = x3[0][:10562]
xdata4 = x4[0][:10562]

ydata1 = run_mean(np.var(x1[1],axis=0),240)
ydata2 = run_mean(np.var(x2[1],axis=0),240)
ydata3 = run_mean(np.var(x3[1],axis=0),240)
ydata4 = run_mean(np.var(x4[1],axis=0),240)

popt1,pcov1 = curve_fit(quad,xdata1[:7000],ydata1[:7000])
popt2,pcov2 = curve_fit(quad,xdata2[:5000],ydata2[:5000])
popt3,pcov3 = curve_fit(quad,xdata3[:3000],ydata3[:3000])
popt4,pcov4 = curve_fit(quad,xdata4[:1000],ydata4[:1000])


plt.figure(figsize=(16,8))

plt.plot (xdata1, ydata1, 'k--',label = 'E=0.05')
plt.plot(xdata1,quad(xdata1,*popt1),'k-', label = 'fit')

plt.plot (xdata2, ydata2,'b--', label = 'E=0.1')
plt.plot(xdata2,quad(xdata2,*popt2),'b-', label = 'fit')

plt.plot (xdata3, ydata3,'m--', label = 'E=0.2')
plt.plot(xdata3,quad(xdata3,*popt3),'m-', label = 'fit')

plt.plot (xdata4, ydata4, 'r--', label = 'E=0.5')
plt.plot(xdata4,quad(xdata4,*popt4),'r-', label = 'fit')

plt.ylabel ('x-coordinate variance')
plt.xlabel ('time, hours')
plt.ylim(-0.1,0.8)
#plt.xlim(-5,150)
plt.title('Variance of x-coordinate of 1000 oscillators with running average (12h,1) fitted to quadratic function', fontsize=16)
plt.legend(loc=1)
plt.show()

###########
FITTING TO LINE
#######

xdata1 = x1[0][:10562]
xdata2 = x2[0][:10562]
xdata3 = x3[0][:10562]
xdata4 = x4[0][:10562]

ydata1 = run_mean(np.var(x1[1],axis=0),240)
ydata2 = run_mean(np.var(x2[1],axis=0),240)
ydata3 = run_mean(np.var(x3[1],axis=0),240)
ydata4 = run_mean(np.var(x4[1],axis=0),240)

popt1,pcov1 = curve_fit(lin,xdata1[:6000],ydata1[:6000])
popt2,pcov2 = curve_fit(lin,xdata2[:4000],ydata2[:4000])
popt3,pcov3 = curve_fit(lin,xdata3[:2000],ydata3[:2000])
popt4,pcov4 = curve_fit(lin,xdata4[:400],ydata4[:400])

plt.figure(figsize=(16,8))
plt.plot (xdata1, ydata1, 'k--',label = 'E=0.05')
plt.plot(xdata1,lin(xdata1,*popt1),'k-', label = 'fit')
plt.plot (xdata2, ydata2,'b--', label = 'E=0.1')
plt.plot(xdata2,lin(xdata2,*popt2),'b-', label = 'fit')
plt.plot (xdata3, ydata3,'m--', label = 'E=0.2')
plt.plot(xdata3,lin(xdata3,*popt3),'m-', label = 'fit')
plt.plot (xdata4, ydata4, 'r--', label = 'E=0.5')
plt.plot(xdata4,lin(xdata4,*popt4),'r-', label = 'fit')

plt.ylabel ('x-coordinate variance')
plt.xlabel ('time, hours')
plt.ylim(-0.1,0.8)
#plt.xlim(-5,150)
plt.title('Variance of x-coordinate of 1000 oscillators with running average (12h,1) fitted to linear function', fontsize=16)
plt.legend(loc=1)
plt.show()


##############
EXPONENTIAL
#############
xdata1 = x1[0][:10562]
xdata2 = x2[0][:10562]
xdata3 = x3[0][:10562]
xdata4 = x4[0][:10562]

ydata1 = run_mean(np.var(x1[1],axis=0),240)
ydata2 = run_mean(np.var(x2[1],axis=0),240)
ydata3 = run_mean(np.var(x3[1],axis=0),240)
ydata4 = run_mean(np.var(x4[1],axis=0),240)

popt1,pcov1 = curve_fit(expon,xdata1,ydata1, maxfev=10000)
popt2,pcov2 = curve_fit(expon,xdata2,ydata2, maxfev=10000)
popt3,pcov3 = curve_fit(expon,xdata3,ydata3, maxfev=10000)
popt4,pcov4 = curve_fit(expon,xdata4,ydata4, maxfev=10000)

#### R-squared statistical metric
rsq=[]
rsq.append(r_sq(expon,xdata1,ydata1,popt1))
rsq.append(r_sq(expon,xdata2,ydata2,popt2))
rsq.append(r_sq(expon,xdata3,ydata3,popt3))
rsq.append(r_sq(expon,xdata4,ydata4,popt4))

plt.figure(figsize=(16,8))

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

plt.plot (xdata1, ydata1, 'k--',label = 'E=0.05')
plt.plot(xdata1,expon(xdata1,*popt1),'k-', label = 'E=0.05 fit')
plt.plot (xdata2, ydata2,'b--', label = 'E=0.1')
plt.plot(xdata2,expon(xdata2,*popt2),'b-', label = 'E=0.1 fit')
plt.plot (xdata3, ydata3,'m--', label = 'E=0.2')
plt.plot(xdata3,expon(xdata3,*popt3),'m-', label = 'E=0.2 fit')
plt.plot (xdata4, ydata4, 'r--', label = 'E=0.5')
plt.plot(xdata4,expon(xdata4,*popt4),'r-', label = 'E=0.5 fit')

plt.ylabel ('x-coordinate variance')
plt.xlabel ('time, hours')
plt.ylim(-0.1,0.8)
#plt.xlim(-5,150)
plt.title('Variance of x-coordinate of 1000 oscillators (smoothened) fitted to exponential', fontsize=16)
plt.legend(loc=1)
plt.show()




##########################################################
#################################################
###########################################        VAR (phase)
#####################################
#################################

plt.figure(figsize=(16,8))

ph1 = phvar(x1[1])[0]
ph2 = phvar(x2[1])[0]
ph3 = phvar(x3[1])[0]
ph4 = phvar(x4[1])[0]

plt.plot(x1[0], ph1, 'k-', label='E=0.05')
plt.plot(x2[0], ph2, 'b-', label='E=0.1')
plt.plot(x3[0], ph3, 'm-', label='E=0.2')
plt.plot(x4[0], ph4, 'r-', label='E=0.5')

plt.xlabel('time, hours')
plt.ylabel('Phase variance')
plt.title('Variance of phase of 1000 oscillators with different noise intensities', fontsize=16)
plt.legend()

########## Smooth

plt.figure(figsize=(16,8))

plt.plot(x1[0][:10562], run_mean(ph1, 240), 'k-', label='E=0.05')
plt.plot(x2[0][:10562], run_mean(ph2, 240), 'b-', label='E=0.1')
plt.plot(x3[0][:10562], run_mean(ph3, 240), 'm-', label='E=0.2')
plt.plot(x4[0][:10562], run_mean(ph4, 240), 'r-', label='E=0.5')

plt.xlabel('time, hours')
plt.ylabel('Phase variance')
plt.title('Variance of phase of 1000 oscillators with different noise intensities', fontsize=16)
plt.legend()



##########
#### LINE FIT
########

xdata1 = np.array(x1[0][:10562])
xdata2 = np.array(x2[0][:10562])
xdata3 = np.array(x3[0][:10562])
xdata4 = np.array(x4[0][:10562])

ydata1 = run_mean(ph1, 240)
ydata2 = run_mean(ph2, 240)
ydata3 = run_mean(ph3, 240)
ydata4 = run_mean(ph4, 240)

popt1,pcov1 = curve_fit(lin,xdata1[:6000],ydata1[:6000], maxfev=10000)
popt2,pcov2 = curve_fit(lin,xdata2[:3000],ydata2[:3000], maxfev=10000)
popt3,pcov3 = curve_fit(lin,xdata3[:700],ydata3[:700], maxfev=10000)
popt4,pcov4 = curve_fit(lin,xdata4[:300],ydata4[:300], maxfev=10000)

plt.figure(figsize=(16,8))
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

plt.plot(xdata1,ydata1, 'k--', label = 'E=0.05')
plt.plot(xdata1, lin(xdata1, *popt1), 'k-', label = 'E=0.05 fit')
plt.plot(xdata2,ydata2, 'b--', label = 'E=0.1')
plt.plot(xdata2, lin(xdata2, *popt2), 'b-', label = 'E=0.1 fit')
plt.plot(xdata3,ydata3, 'm--', label = 'E=0.2')
plt.plot(xdata3, lin(xdata3, *popt3),'m-', label = 'E=0.2 fit')
plt.plot(xdata4,ydata4, 'r--', label = 'E=0.5')
plt.plot(xdata4, lin(xdata4, *popt4), 'r-', label = 'E=0.5 fit')

plt.xlabel('time, hours')
plt.ylabel('Phase variance')
plt.xlim(-20,350)
plt.ylim(-100,2800)
plt.title('Variance of phase of 1000 noisy oscillators fitted to line', fontsize=16)
plt.legend(loc=1)


#########
### QUADRATIC
####

xdata1 = np.array(x1[0][:10562])
xdata2 = np.array(x2[0][:10562])
xdata3 = np.array(x3[0][:10562])
xdata4 = np.array(x4[0][:10562])

ydata1 = run_mean(ph1, 240)
ydata2 = run_mean(ph2, 240)
ydata3 = run_mean(ph3, 240)
ydata4 = run_mean(ph4, 240)

popt1,pcov1 = curve_fit(quad,xdata1[:6000],ydata1[:6000], maxfev=10000)
popt2,pcov2 = curve_fit(quad,xdata2[:3000],ydata2[:3000], maxfev=10000)
popt3,pcov3 = curve_fit(quad,xdata3[:700],ydata3[:700], maxfev=10000)
popt4,pcov4 = curve_fit(quad,xdata4[:300],ydata4[:300], maxfev=10000)

plt.figure(figsize=(16,8))
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

plt.plot(xdata1,ydata1, 'k--', label = 'E=0.05')
plt.plot(xdata1, quad(xdata1, *popt1), 'k-', label = 'E=0.05 fit')
plt.plot(xdata2,ydata2, 'b--', label = 'E=0.1')
plt.plot(xdata2, quad(xdata2, *popt2), 'b-', label = 'E=0.1 fit')
plt.plot(xdata3,ydata3, 'm--', label = 'E=0.2')
plt.plot(xdata3, quad(xdata3, *popt3),'m-', label = 'E=0.2 fit')
plt.plot(xdata4,ydata4, 'r--', label = 'E=0.5')
plt.plot(xdata4, quad(xdata4, *popt4), 'r-', label = 'E=0.5 fit')

plt.xlabel('time, hours')
plt.ylabel('Phase variance')
plt.xlim(-20,350)
plt.ylim(-100,2800)
plt.title('Variance of phase of 1000 noisy oscillators fitted to quadratic', fontsize=16)
plt.legend(loc=1)



########################

EXPONENTIAL

#######################

xdata1 = np.array(x1[0][:10562])
xdata2 = np.array(x2[0][:10562])
xdata3 = np.array(x3[0][:10562])
xdata4 = np.array(x4[0][:10562])

ydata1 = run_mean(ph1, 240)
ydata2 = run_mean(ph2, 240)
ydata3 = run_mean(ph3, 240)
ydata4 = run_mean(ph4, 240)

#popt1,pcov1 = curve_fit(expon, xdata1[:6000], ydata1[:6000], maxfev=10000)
#popt2,pcov2 = curve_fit(expon, xdata2[:3000], ydata2[:3000], maxfev=10000)
#popt3,pcov3 = curve_fit(expon, xdata3[:700], ydata3[:700], maxfev=10000)
#popt4,pcov4 = curve_fit(expon, xdata4[:300], ydata4[:300], maxfev=10000)

popt1 = [-4838.26, 0.000317081, 4856.2] #R^2 = 0.9994
popt2 = [-2840, 0.0022, 2900] #R2 = 0.9984 # This one I fitted almost manually

popt3 = [-2592.47, 0.0109569, 2708.4]   #R^2 = 0.9993
popt4 = [-1828.32, 0.0711882, 2660.94]  #R^2 = 0.9978

# Coefficients and R2 are from Desmos.com
# Using ydata4Desmos = [ydata4[i] for i in range(2000) if i%50==0]
# xdata4Desmos = [xdata4[i] for i in range(2000) if i%50==0]
# ydata3Desmos = [ydata3[i] for i in range(8000) if i%170==0]
# xdata3Desmos = [xdata3[i] for i in range(8000) if i%170==0]
# ydata1Desmos = [ydata1[i] for i in range(9000) if i%190==0]
# xdata1Desmos = [xdata1[i] for i in range(9000) if i%190==0]


plt.figure(figsize=(16,8))
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

plt.plot(xdata1,ydata1, 'k--', label = 'E=0.05')
plt.plot(xdata1, expon(xdata1, *popt1), 'k-', label = 'E=0.05 fit')
plt.plot(xdata2,ydata2, 'b--', label = 'E=0.1')
plt.plot(xdata2, expon(xdata2, *popt2), 'b-', label = 'E=0.1 fit')
plt.plot(xdata3,ydata3, 'm--', label = 'E=0.2')
plt.plot(xdata3, expon(xdata3, *popt3),'m-', label = 'E=0.2 fit')
plt.plot(xdata4,ydata4, 'r--', label = 'E=0.5')
plt.plot(xdata4, expon(xdata4, *popt4), 'r-', label = 'E=0.5 fit')

plt.xlabel('time, hours')
plt.ylabel('Phase variance')
plt.xlim(-20,350)
plt.ylim(-100,2800)
plt.title('Variance of phase of 1000 noisy oscillators fitted to exponential', fontsize=16)
plt.legend(loc=1)














"""















