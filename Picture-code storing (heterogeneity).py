#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 15:18:52 2018

@author: kalashnikov
"""

"""
IN THIS FILE I WILL POST ONLY PARTS OF THE CODE 
DEVOTED TO PLOTTING, USING FUNCTIONS FROM n_coupled_oscillators.py
"""

"""

##############################
# STARTING THE ENGINE
#############################
n = 1000
t = np.linspace(0, 400, 4000)
state0 = [1,0]*n
# Example of the execution:
# x51 = odeint(oscillator_system, state05, t5, args = (([0.1]*n,[1]*n,[(np.pi*2)/(24 + 0.5*i) for i in np.random.randn(n)],[0.0]*n,[0.0]*n, [0.0]*n)))


x1 = np.load("/home/kalashnikov/Code/Variables for my code/Heterogeneity/Other state0/1000 oscillators with sigma 0.5 and state0 [1,0].npy")
x2 = np.load("/home/kalashnikov/Code/Variables for my code/Heterogeneity/Other state0/1000 oscillators with sigma 1 and state0 [1,0].npy")
x3 = np.load("/home/kalashnikov/Code/Variables for my code/Heterogeneity/Other state0/1000 oscillators with sigma 1.5 and state0 [1,0].npy")
x4 = np.load("/home/kalashnikov/Code/Variables for my code/Heterogeneity/Other state0/1000 oscillators with sigma 2 and state0 [1,0].npy")

x1x = sep(x1)[0]
x2x = sep(x2)[0]
x3x = sep(x3)[0]
x4x = sep(x4)[0]


def me4(x):
    return maxs3(extr(x))


# Default parameters
# 
font = {'family' : 'normal',
        'weight' : 'bold'}
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

"""


"""
################################################## HETEROGENEITY
##################################################

###### MEAN(X) VS. TIME
plt.figure(figsize=(16,8))

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

plt.plot (t, np.mean(x1x,axis=0),'k-', label = 'sigma = 0.5')
plt.plot (t, np.mean(x2x,axis=0),'b-', label = 'sigma = 1')
plt.plot (t, np.mean(x3x,axis=0),'m-', label = 'sigma = 1.5')
plt.plot (t, np.mean(x4x,axis=0),'r-', label = 'sigma = 2')

plt.ylabel ('x-coordinate')
plt.xlabel ('time, hours')
#plt.ylim(-1.5,2.5)
plt.xlim(-10,350)
plt.title('Mean (x-coordinate) of 1000 heterogenous oscillators', fontsize=16)
plt.legend()
plt.show()



########## SAME, but printable

plt.figure(figsize=(14,10))

plt.rc('font', size=16)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=22)  # fontsize of the figure title

plt.plot (t, np.mean(x1x,axis=0),'k-', label = 'sigma=0.5')
plt.plot (t, np.mean(x2x,axis=0),'b-', label = 'sigma=1')
plt.plot (t, np.mean(x3x,axis=0),'m-', label = 'sigma=1.5')
plt.plot (t, np.mean(x4x,axis=0),'r-', label = 'sigma=2')

plt.ylabel ('x-coordinate')
plt.xlabel ('time, hours')
#plt.ylim(-1.5,2.5)
plt.xlim(-10,400)
plt.title('Mean (x-coordinate) of 1000 heterogenous oscillators', fontsize=22)
plt.legend()
plt.show()




###### MEAN(X) AND MAXIMA VS. TIME 

plt.figure(figsize=(14,10))

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

plt.plot(t,np.mean(x1x,axis=0),'k-', label="sigma = 0.5")
plt.plot(t,np.mean(x2x,axis=0),'b-', label="sigma =  1.0")
plt.plot(t,np.mean(x3x,axis=0),'m-', label="sigma =  1.5")
plt.plot(t,np.mean(x4x,axis=0),'r-', label="sigma =  2.0")

plt.plot(me4(np.mean(x1x,axis=0))[0], me4(np.mean(x1x,axis=0))[1], 'ko', label="maxima sigma 0.5")
plt.plot(me4(np.mean(x2x,axis=0))[0], me4(np.mean(x2x,axis=0))[1], 'bo', label="maxima sigma 1.0")
plt.plot(me4(np.mean(x3x,axis=0))[0], me4(np.mean(x3x,axis=0))[1], 'mo', label="maxima sigma 1.5")
plt.plot(me4(np.mean(x4x,axis=0))[0], me4(np.mean(x4x,axis=0))[1], 'ro', label="maxima sigma 2.0")

plt.ylabel ('x-coordinate')
plt.xlabel ('time, hours')
plt.xlim(-10,350)
plt.title('Mean and its maxima (x-coordinate) of 1000 heterogenous oscillators', fontsize=16)
plt.legend()


###################
##### FITTING TO LINEAR


xdata1 = np.array(me4(np.mean(x1x, axis=0))[0])
ydata1 = np.array(me4(np.mean(x1x, axis=0))[1])
xdata2 = np.array(me4(np.mean(x2x, axis=0))[0])
ydata2 = np.array(me4(np.mean(x2x, axis=0))[1])
xdata3 = np.array(me4(np.mean(x3x, axis=0))[0])
ydata3 = np.array(me4(np.mean(x3x, axis=0))[1])
xdata4 = np.array(me4(np.mean(x4x, axis=0))[0])
ydata4 = np.array(me4(np.mean(x4x, axis=0))[1])

popt1,pcov1 = curve_fit(lin,xdata1[3:10],ydata1[3:10])
popt2,pcov2 = curve_fit(lin,xdata2[2:8],ydata2[2:8])
popt3,pcov3 = curve_fit(lin,xdata3[1:5],ydata3[1:5])
popt4,pcov4 = curve_fit(lin,xdata4[1:4],ydata4[1:4])
tangents = [popt1[0],popt2[0],popt3[0], popt4[0]]
rt = roundl(tangents,4)
nt = roundl((rt/rt[0]).tolist(),4)

# R-squared metric
rsq = []
rsq.append(r_sq(lin,xdata1[3:10],ydata1[3:10],popt1))
rsq.append(r_sq(lin,xdata2[2:8],ydata2[2:8],popt2))
rsq.append(r_sq(lin,xdata3[1:5],ydata3[1:5],popt3))
rsq.append(r_sq(lin,xdata4[1:4],ydata4[1:4],popt4))


plt.figure(figsize=(14,10))

plt.rc('font', size=10)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=11.5)    # legend fontsize
plt.rc('figure', titlesize=15)  # fontsize of the figure title

plt.plot(xdata1,ydata1, 'ko', label = 's=0.5 maxima')
plt.plot(xdata1, lin(xdata1, *popt1), 'k--', label = 's=0.5 fit')
plt.plot(xdata2,ydata2, 'bo', label = 's=1.0 maxima')
plt.plot(xdata2, lin(xdata2, *popt2), 'b--', label = 's=1.0 fit')
plt.plot(xdata3,ydata3, 'mo', label = 's=1.5 maxima')
plt.plot(xdata3, lin(xdata3, *popt3),'m--', label = 's=1.5 fit')
plt.plot(xdata4,ydata4, 'ro', label = 's=2.0 maxima')
plt.plot(xdata4, lin(xdata4, *popt4), 'r--', label = 's=2.0 fit')

#plt.plot(np.linspace(0,400,400), [0.23]*400, '-', label = 'Lowest cut-off limit')
#plt.plot(np.linspace(0,180,180), [0.945]*180, '-', label = 'Highest cut-off limit')

plt.ylabel ('x-coordinate')
plt.xlabel ('time, hours')
texttang = ''
texttang2 = ''
for i in rt:
    texttang = texttang+str(i)+' : '
for i in nt:
    texttang2 = texttang2+str(i)+' : '
texttang = texttang[:-3]
texttang2 = texttang2[:-3]
plt.text(90,1.1,'Slopes\n' + texttang)
plt.text(90,1.0,'Normalized\n' + texttang2)

plt.xlim(-5,230)
plt.ylim(-0.2,1.2)
plt.title('Mean (x-coordinate) of 1000 heterogenous oscillators fitted to the line', fontsize=16)
plt.legend()


########
Exponential fit
####


# From desmos.com

xdata1 = np.array(me4(np.mean(x1x, axis=0))[0])
ydata1 = np.array(me4(np.mean(x1x, axis=0))[1])
xdata2 = np.array(me4(np.mean(x2x, axis=0))[0])
ydata2 = np.array(me4(np.mean(x2x, axis=0))[1])
xdata3 = np.array(me4(np.mean(x3x, axis=0))[0])
ydata3 = np.array(me4(np.mean(x3x, axis=0))[1])
xdata4 = np.array(me4(np.mean(x4x, axis=0))[0])
ydata4 = np.array(me4(np.mean(x4x, axis=0))[1])

#popt1,pcov1 = curve_fit(expon,xdata1[3:10],ydata1[3:10], maxfev=10000)
#popt2,pcov2 = curve_fit(expon,xdata2[2:8],ydata2[2:8], maxfev=10000)
#popt3,pcov3 = curve_fit(expon,xdata3[1:5],ydata3[1:5], maxfev=10000)
#popt4,pcov4 = curve_fit(expon,xdata4[1:4],ydata4[1:4], maxfev=10000)

popt1 = [700.76, 0.00000393338, -699.683] #R^2=0.9879
popt2 = [1.23586, 0.00727099, -0.107034] #R^2=0.9632
popt3 = [1.11169, 0.0120443, -0.00746322] #R^2=0.9699
popt4 = [1.08954, 0.0158688, -0.00718665] #R2 = 0.9743

plt.figure(figsize=(14,10))

plt.rc('font', size=10)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=11.5)    # legend fontsize
plt.rc('figure', titlesize=15)  # fontsize of the figure title

plt.plot(xdata1,ydata1, 'ko', label = 's=0.5 maxima')
plt.plot(xdata1, expon(xdata1, *popt1), 'k--', label = 's=0.5 fit')
plt.plot(xdata2,ydata2, 'bo', label = 's=1.0 maxima')
plt.plot(xdata2, expon(xdata2, *popt2), 'b--', label = 's=1.0 fit')
plt.plot(xdata3,ydata3, 'mo', label = 's=1.5 maxima')
plt.plot(xdata3, expon(xdata3, *popt3),'m--', label = 's=1.5 fit')
plt.plot(xdata4,ydata4, 'ro', label = 's=2.0 maxima')
plt.plot(xdata4, expon(xdata4, *popt4), 'r--', label = 's=2.0 fit')

#plt.plot(np.linspace(0,400,400), [0.23]*400, '-', label = 'Lowest cut-off limit')
#plt.plot(np.linspace(0,180,180), [0.945]*180, '-', label = 'Highest cut-off limit')

plt.ylabel ('x-coordinate')
plt.xlabel ('time, hours')

plt.xlim(-5,230)
plt.ylim(-0.2,1.2)
plt.title('Mean (x-coordinate) of 1000 heterogenous oscillators fitted to the exponential', fontsize=16)
plt.legend()



#########
QUADRATIC
########
popt1,pcov1 = curve_fit(quad,xdata1,ydata1, maxfev=10000)
popt2,pcov2 = curve_fit(quad,xdata2,ydata2, maxfev=10000)
popt3,pcov3 = curve_fit(quad,xdata3,ydata3, maxfev=10000)
popt4,pcov4 = curve_fit(quad,xdata4,ydata4, maxfev=10000)

#popt1 = [700.76, 0.00000393338, -699.683] #R^2=0.9879
#popt2 = [1.23586, 0.00727099, -0.107034] #R^2=0.9632
#popt3 = [1.11169, 0.0120443, -0.00746322] #R^2=0.9699
#popt4 = [1.08954, 0.0158688, -0.00718665] #R2 = 0.9743

plt.figure(figsize=(14,10))

plt.plot(xdata1,ydata1, 'ko', label = 's=0.5 maxima')
plt.plot(xdata1, quad(xdata1, *popt1), 'k--', label = 's=0.5 fit')
plt.plot(xdata2,ydata2, 'bo', label = 's=1.0 maxima')
plt.plot(xdata2, quad(xdata2, *popt2), 'b--', label = 's=1.0 fit')
plt.plot(xdata3,ydata3, 'mo', label = 's=1.5 maxima')
plt.plot(xdata3, quad(xdata3, *popt3),'m--', label = 's=1.5 fit')
plt.plot(xdata4,ydata4, 'ro', label = 's=2.0 maxima')
plt.plot(xdata4, quad(xdata4, *popt4), 'r--', label = 's=2.0 fit')

#plt.plot(np.linspace(0,400,400), [0.23]*400, '-', label = 'Lowest cut-off limit')
#plt.plot(np.linspace(0,180,180), [0.945]*180, '-', label = 'Highest cut-off limit')

plt.ylabel ('x-coordinate')
plt.xlabel ('time, hours')

#plt.xlim(-5,230)
plt.ylim(-0.2,1.2)
plt.title('Mean (x-coordinate) of 1000 heterogenous oscillators fitted to the quadratic', fontsize=16)
plt.legend(loc=1)








##############################
######  VAR (X)
###########

plt.figure(figsize=(14,10))

plt.plot (t, np.var(x1x,axis=0), label = 'sigma=0.5')
plt.plot (t, np.var(x2x,axis=0), label = 'sigma=1')
plt.plot (t, np.var(x3x,axis=0), label = 'sigma=1.5')
plt.plot (t, np.var(x4x,axis=0), label = 'sigma=2')

plt.ylabel ('x-coordinate variance')
plt.xlabel ('time, hours')

plt.ylim(-0.05,0.65)
plt.title ('Variance of x-coordinate of 1000 heterogenous oscillators (raw)', fontsize=16)
plt.legend()
plt.show()

######## VAR (X) (run_mean)

plt.figure(figsize=(14,10))

plt.rc('font', size=10)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=11.5)    # legend fontsize
plt.rc('figure', titlesize=15)  # fontsize of the figure title

plt.plot (t[:3762], run_mean(np.var(x1x,axis=0),120,1), 'k-', label = 'sigma=0.5')
plt.plot (t[:3762], run_mean(np.var(x2x,axis=0),120,1), 'b-', label = 'sigma=1')
plt.plot (t[:3762], run_mean(np.var(x3x,axis=0),120,1), 'm-', label = 'sigma=1.5')
plt.plot (t[:3762], run_mean(np.var(x4x,axis=0),120,1), 'r-', label = 'sigma=2')

plt.ylabel ('x-coordinate variance')
plt.xlabel ('time, hours')
plt.title ('Variance of x-coordinate of 1000 heterogenous oscillators (smoothened)', fontsize=16)
plt.xlim(-10,250)
plt.legend()
plt.show()



#############
##### FITTING TO QUAD()

xdata=t[:3762]
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

plt.figure(figsize=(14,10))
plt.plot (xdata, ydata1, 'k--',label = 'sigma=0.5')
plt.plot(xdata,quad(xdata,*popt1),'k-', label = 'fit')
plt.plot (xdata, ydata2,'b--', label = 'sigma=1')
plt.plot(xdata,quad(xdata,*popt2),'b-', label = 'fit')
plt.plot (xdata, ydata3,'m--', label = 'sigma=1.5')
plt.plot(xdata,quad(xdata,*popt3),'m-', label = 'fit')
plt.plot (xdata, ydata4, 'r--', label = 'sigma=2')
plt.plot(xdata,quad(xdata,*popt4),'r-', label = 'fit')

plt.ylabel ('x-coordinate variance')
plt.xlabel ('time, hours')
plt.ylim(-0.05,0.55)
plt.xlim(-5,150)

texttang = ''
for i in ct:
    texttang = texttang+str(i)+' : '
texttang = texttang[:-3]
plt.text(100,0,'The first coefficients of curves are\n' + texttang)
#plt.yscale('log')
plt.title('Variance (x-coordinate) of 1000 heterogenous oscillators fitted to quadratic', fontsize=16)
plt.legend()
plt.show()




##########
To exponential - from Desmos
######
#xdata0Desmos = [xdata[i] for i in range(3762) if i%100==0]
#ydata1Desmos = [ydata1[i] for i in range(3762) if i%100==0]
#ydata2Desmos = [ydata2[i] for i in range(3762) if i%100==0]
#ydata3Desmos = [ydata3[i] for i in range(3762) if i%100==0]
#ydata4Desmos = [ydata4[i] for i in range(3762) if i%100==0]

#popt1 = [-0.768452, 0.00391461, 0.714247] #R^2=0.9828
#popt2 = [-0.576554, 0.0135749, 0.520838] #R^2=0.9773
#popt3 = [-0.536825, 0.0224373, 0.50659] #R^2=0.9803
#popt4 = [-0.495557, 0.030814, 0.503681] #R^2=0.9878

## Took the first 3-4 points from the tables
popt1 = [-0.746697, 0.00543704, 0.623379]
popt2 = [-0.760352, 0.0182613, 0.5087]
popt3 = [-0.719447, 0.0293259, 0.501899]
popt4 = [-0.691959, 0.0408108, 0.500804]

plt.figure(figsize=(14,10))
plt.plot (xdata, ydata1, 'k--',label = 'sigma=0.5')
plt.plot(xdata,expon(xdata,*popt1),'k-', label = 'fit')
plt.plot (xdata, ydata2,'b--', label = 'sigma=1')
plt.plot(xdata,expon(xdata,*popt2),'b-', label = 'fit')
plt.plot (xdata, ydata3,'m--', label = 'sigma=1.5')
plt.plot(xdata,expon(xdata,*popt3),'m-', label = 'fit')
plt.plot (xdata, ydata4, 'r--', label = 'sigma=2')
plt.plot(xdata,expon(xdata,*popt4),'r-', label = 'fit')

plt.ylabel ('x-coordinate variance')
plt.xlabel ('time, hours')
plt.ylim(-0.05,0.55)
plt.xlim(-5,150)

#plt.yscale('log')
plt.title('Variance (x-coordinate) of 1000 heterogenous oscillators fitted to exponential', fontsize=16)
plt.legend()
plt.show()

################
######### VAR (PHASE)
################

plt.figure(figsize=(14,10))
plt.plot (t, phvar(x1)[0],'k-', label = 'sigma=0.5')
plt.plot (t, phvar(x2)[0],'b-', label = 'sigma=1')
plt.plot (t, phvar(x3)[0],'m-', label = 'sigma=1.5')
plt.plot (t, phvar(x4)[0],'r-', label = 'sigma=2')

plt.ylabel ('Variance of phase')
plt.xlabel ('time, hours')
plt.title('Variance of phase of 1000 heterogenous oscillators', fontsize=16)
plt.legend()
plt.show()

###### run_mean(VAR (PHASE))

plt.figure(figsize=(14,10))
plt.plot (t[:3762], run_mean(phvar(x1)[0],120,1),'k-', label = 'sigma=0.5')
plt.plot (t[:3762], run_mean(phvar(x2)[0],120,1),'b-', label = 'sigma=1')
plt.plot (t[:3762], run_mean(phvar(x3)[0],120,1),'m-', label = 'sigma=1.5')
plt.plot (t[:3762], run_mean(phvar(x4)[0],120,1),'r-', label = 'sigma=2')

plt.ylabel ('Variance of phase')
plt.xlabel ('time, hours')
plt.title('Variance of phase of 1000 heterogenous oscillators (smoothened)', fontsize=16)
plt.legend()
plt.show()


#####################
##### FITTING TO LINEAR
xdata=t[:3762]
ydata1=run_mean(phvar(x1)[0],120,1)
ydata2=run_mean(phvar(x2)[0],120,1)
ydata3=run_mean(phvar(x3)[0],120,1)
ydata4=run_mean(phvar(x4)[0],120,1)

popt1,pcov1 = curve_fit(lin,xdata[40:2000],ydata1[40:2000])
popt2,pcov2 = curve_fit(lin,xdata[:800],ydata2[:800])
popt3,pcov3 = curve_fit(lin,xdata[:700],ydata3[:700])
popt4,pcov4 = curve_fit(lin,xdata[:500],ydata4[:500])
tangents = [popt1[0],popt2[0],popt3[0], popt4[0]]
rt = roundl(tangents,3)
nt = roundl((rt/rt[0]).tolist(),3)

plt.figure(figsize=(14,10))

plt.plot (xdata, ydata1, 'k--',label = 'sigma=0.5 data')
plt.plot(xdata,lin(xdata,*popt1),'k-', label = 'fit')
plt.plot (xdata, ydata2,'b--', label = 'sigma=1')
plt.plot(xdata,lin(xdata,*popt2),'b-', label = 'fit')
plt.plot (xdata, ydata3,'m--', label = 'sigma=1.5')
plt.plot(xdata,lin(xdata,*popt3),'m-', label = 'fit')
plt.plot (xdata, ydata4, 'r--', label = 'sigma=2')
plt.plot(xdata,lin(xdata,*popt4),'r-', label = 'fit')

plt.ylabel ('Variance of phase')
plt.xlabel ('time, hours')
plt.ylim(-50,3000)

texttang = ''
texttang2 = ''
for i in rt:
    texttang = texttang+str(i)+' : '
for i in nt:
    texttang2 = texttang2+str(i)+' : '
texttang = texttang[:-3]
texttang2 = texttang2[:-3]
plt.text(200,1000,'Slopes\n' + texttang)
plt.text(200,500,'Normalized\n' + texttang2)

plt.title ('Variance of phase of 1000 heterogenous oscillators (smoothened) fitted to linear function', fontsize=16)
plt.legend()
plt.show()


##################
######  FITTING TO QUAD
###########

xdata=t[:3762]
ydata1=run_mean(phvar(x1)[0],120,1)
ydata2=run_mean(phvar(x2)[0],120,1)
ydata3=run_mean(phvar(x3)[0],120,1)
ydata4=run_mean(phvar(x4)[0],120,1)

popt1,pcov1 = curve_fit(quad,xdata[:800],ydata1[:800])
popt2,pcov2 = curve_fit(quad,xdata[:400],ydata2[:400])
popt3,pcov3 = curve_fit(quad,xdata[:300],ydata3[:300])
popt4,pcov4 = curve_fit(quad,xdata[:200],ydata4[:200])

coefficients = [popt1[0],popt2[0],popt3[0], popt4[0]]
ct = roundl(coefficients,3)

plt.figure(figsize=(14,10))

plt.plot (xdata, ydata1, 'r--',label = 'sigma=0.5')
plt.plot(xdata,quad(xdata,*popt1),'r-', label = 'fit')
plt.plot (xdata, ydata2,'m--', label = 'sigma=1')
plt.plot(xdata,quad(xdata,*popt2),'m-', label = 'fit')
plt.plot (xdata, ydata3,'b--', label = 'sigma=1.5')
plt.plot(xdata,quad(xdata,*popt3),'b-', label = 'fit')
plt.plot (xdata, ydata4, 'k--', label = 'sigma=2')
plt.plot(xdata,quad(xdata,*popt4),'k-', label = 'fit')

plt.ylabel ('Variance of phase')
plt.xlabel ('time, hours')
plt.ylim(-50,3000)
#plt.yscale('log')
#plt.xscale('log')
plt.xlim(-5,150)

texttang = ''
for i in ct:
    texttang = texttang+str(i)+' : '
texttang = texttang[:-3]
plt.text(60,50,'The coefficients of curves are\n' + texttang)
plt.title ('Variance of phase of 1000 heterogenous oscillators (smoothened) fitted to quadratic', fontsize=16)
plt.legend()
plt.show()


#########
EXPONENTIAL
#########

popt1,pcov1 = curve_fit(expon,xdata,ydata1,maxfev=10000)
popt2,pcov2 = curve_fit(expon,xdata,ydata2,maxfev=10000)
popt3,pcov3 = curve_fit(expon,xdata,ydata3,maxfev=10000)
popt4,pcov4 = curve_fit(expon,xdata,ydata4,maxfev=10000)

plt.figure(figsize=(14,10))

plt.plot (xdata, ydata1, 'r--',label = 'sigma=0.5')
plt.plot(xdata,expon(xdata,*popt1),'r-', label = 'fit')
plt.plot (xdata, ydata2,'m--', label = 'sigma=1')
plt.plot(xdata,expon(xdata,*popt2),'m-', label = 'fit')
plt.plot (xdata, ydata3,'b--', label = 'sigma=1.5')
plt.plot(xdata,expon(xdata,*popt3),'b-', label = 'fit')
plt.plot (xdata, ydata4, 'k--', label = 'sigma=2')
plt.plot(xdata,expon(xdata,*popt4),'k-', label = 'fit')

plt.ylabel ('Variance of phase')
plt.xlabel ('time, hours')
plt.ylim(-50,3000)
#plt.yscale('log')
#plt.xscale('log')
#plt.xlim(-5,150)

plt.title ('Variance of phase of 1000 heterogenous oscillators (smoothened) fitted to exponential', fontsize=16)
plt.legend()
plt.show()




















########################################################################################################
########################################################

MEAN (maybe only change the legend?)
############
font = {'style' : 'normal',
        'weight' : 'normal',
        'family' : 'DejaVu Sans'}
plt.rc('font', **font)      
plt.figure(figsize=(14,10))

plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=22)     # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=24)    # fontsize of the tick labels
plt.rc('ytick', labelsize=24)    # fontsize of the tick labels
plt.rc('legend', fontsize=26)    # legend fontsize
plt.rc('figure', titlesize=26)  # fontsize of the figure title

plt.plot (t, np.mean(x1x,axis=0),'k-', label = 'sigma = 0.5')
plt.plot (t, np.mean(x2x,axis=0),'b-', label = 'sigma = 1')
plt.plot (t, np.mean(x3x,axis=0),'m-', label = 'sigma = 1.5')
plt.plot (t, np.mean(x4x,axis=0),'r-', label = 'sigma = 2')

plt.ylabel ('x-coordinate')
plt.xlabel ('time, hours')
#plt.ylim(-1.5,2.5)
plt.xlim(-10,400)
plt.title('Mean (x-coordinate) of 1000 heterogenous oscillators', fontsize=26)
plt.legend()
plt.show()




####
MEAN LINE FIT
####

xdata1 = np.array(me4(np.mean(x1x, axis=0))[0])
ydata1 = np.array(me4(np.mean(x1x, axis=0))[1])
xdata2 = np.array(me4(np.mean(x2x, axis=0))[0])
ydata2 = np.array(me4(np.mean(x2x, axis=0))[1])
xdata3 = np.array(me4(np.mean(x3x, axis=0))[0])
ydata3 = np.array(me4(np.mean(x3x, axis=0))[1])
xdata4 = np.array(me4(np.mean(x4x, axis=0))[0])
ydata4 = np.array(me4(np.mean(x4x, axis=0))[1])

popt1,pcov1 = curve_fit(lin,xdata1[3:10],ydata1[3:10])
popt2,pcov2 = curve_fit(lin,xdata2[2:8],ydata2[2:8])
popt3,pcov3 = curve_fit(lin,xdata3[1:5],ydata3[1:5])
popt4,pcov4 = curve_fit(lin,xdata4[1:4],ydata4[1:4])
tangents = [popt1[0],popt2[0],popt3[0], popt4[0]]
rt = roundl(tangents,4)
nt = roundl((rt/rt[0]).tolist(),4)

# R-squared metric
rsq = []
rsq.append(r_sq(lin,xdata1[3:10],ydata1[3:10],popt1))
rsq.append(r_sq(lin,xdata2[2:8],ydata2[2:8],popt2))
rsq.append(r_sq(lin,xdata3[1:5],ydata3[1:5],popt3))
rsq.append(r_sq(lin,xdata4[1:4],ydata4[1:4],popt4))

plt.figure(figsize=(14,10))
#plt.rc('legend', fontsize=22)    # legend fontsize
plt.plot(xdata1,ydata1, 'ko', label = 's=0.5 maxima')
plt.plot(xdata1, lin(xdata1, *popt1), 'k--', label = 's=0.5 fit')
plt.plot(xdata2,ydata2, 'bo', label = 's=1.0 maxima')
plt.plot(xdata2, lin(xdata2, *popt2), 'b--', label = 's=1.0 fit')
plt.plot(xdata3,ydata3, 'mo', label = 's=1.5 maxima')
plt.plot(xdata3, lin(xdata3, *popt3),'m--', label = 's=1.5 fit')
plt.plot(xdata4,ydata4, 'ro', label = 's=2.0 maxima')
plt.plot(xdata4, lin(xdata4, *popt4), 'r--', label = 's=2.0 fit')

#plt.plot(np.linspace(0,400,400), [0.23]*400, '-', label = 'Lowest cut-off limit')
#plt.plot(np.linspace(0,180,180), [0.945]*180, '-', label = 'Highest cut-off limit')

plt.ylabel ('x-coordinate')
plt.xlabel ('time, hours')
texttang = ''
texttang2 = ''
for i in rt:
    texttang = texttang+str(i)+' : '
for i in nt:
    texttang2 = texttang2+str(i)+' : '
texttang = texttang[:-3]
texttang2 = texttang2[:-3]
#plt.text(90,1.1,'Slopes\n' + texttang)
#plt.text(90,1.0,'Normalized\n' + texttang2)

plt.xlim(-15,400)
plt.ylim(-0.2,1.2)
plt.title('Mean (x-coord) of heterogenous oscillators fitted to the line', fontsize=26)
plt.legend()



















"""
