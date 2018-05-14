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

plt.plot (t, np.mean(x1x,axis=0),'k-', label = 'sigma=0.5')
plt.plot (t, np.mean(x2x,axis=0),'b-', label = 'sigma=1')
plt.plot (t, np.mean(x3x,axis=0),'m-', label = 'sigma=1.5')
plt.plot (t, np.mean(x4x,axis=0),'r-', label = 'sigma=2')

plt.ylabel ('x-coordinate')
plt.xlabel ('time, hours')
#plt.ylim(-1.5,2.5)
plt.xlim(-10,350)
plt.title('Mean (x-coordinate) of 1000 heterogenous oscillators', fontsize=16)
plt.legend()
plt.show()



###### MEAN(X) AND MAXIMA VS. TIME 

plt.figure(figsize=(16,8))

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

plt.plot(t,np.mean(x1x,axis=0),'k-', label="sigma 0.5")
plt.plot(t,np.mean(x2x,axis=0),'b-', label="sigma 1.0")
plt.plot(t,np.mean(x3x,axis=0),'m-', label="sigma 1.5")
plt.plot(t,np.mean(x4x,axis=0),'r-', label="sigma 2.0")

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

plt.figure(figsize=(11,8))

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


##################################
######  VAR (X)


plt.figure(figsize=(16,8))

plt.plot (t, np.var(x1x,axis=0), label = 'sigma=0.5')
plt.plot (t, np.var(x2x,axis=0), label = 'sigma=1')
plt.plot (t, np.var(x3x,axis=0), label = 'sigma=1.5')
plt.plot (t, np.var(x4x,axis=0), label = 'sigma=2')

plt.ylabel ('Variance of x-coordinate of 1000 oscillators')
plt.xlabel ('time, hours')

plt.ylim(-0.05,0.65)
plt.legend()
plt.show()

######## VAR (X) (run_mean)

plt.figure(figsize=(12,8))

plt.rc('font', size=10)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=11.5)    # legend fontsize
plt.rc('figure', titlesize=15)  # fontsize of the figure title

plt.plot (t[:3762], run_mean(np.var(x1x,axis=0),120,1), label = 'sigma=0.5')
plt.plot (t[:3762], run_mean(np.var(x2x,axis=0),120,1), label = 'sigma=1')
plt.plot (t[:3762], run_mean(np.var(x3x,axis=0),120,1), label = 'sigma=1.5')
plt.plot (t[:3762], run_mean(np.var(x4x,axis=0),120,1), label = 'sigma=2')

plt.ylabel ('x-coordinate')
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

plt.figure(figsize=(12,10))
plt.plot (xdata, ydata1, 'r--',label = 'sigma=0.5')
plt.plot(xdata,quad(xdata,*popt1),'r-', label = 'fit')
plt.plot (xdata, ydata2,'m--', label = 'sigma=1')
plt.plot(xdata,quad(xdata,*popt2),'m-', label = 'fit')
plt.plot (xdata, ydata3,'b--', label = 'sigma=1.5')
plt.plot(xdata,quad(xdata,*popt3),'b-', label = 'fit')
plt.plot (xdata, ydata4, 'k--', label = 'sigma=2')
plt.plot(xdata,quad(xdata,*popt4),'k-', label = 'fit')

plt.ylabel ('x-coordinate')
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


"""
