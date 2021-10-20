# Python 範例 --- 波動篇第三週
#範例三，傅立葉級數，波的組合

import numpy as np
from matplotlib import pyplot as plt

x_start=0
x_stop=4

L0=2

ax = plt.axes()
ax.axis([x_start,x_stop,-1.5,1.5])
ax.set_xlabel('x')
ax.set_ylabel('f')


x = np.arange(x_start,x_stop, 0.01)

NWaves=8
a=np.zeros(NWaves)
b=np.zeros(NWaves)
f = np.zeros([NWaves,x.shape[0]])
fsum = np.zeros(x.shape[0])
f0 = np.zeros(x.shape[0])


# 方波係數

b[1]=1.27
b[3]=0.42
b[5]=0.25
b[7]=0.18
for i in range(0,x.shape[0]): 
    if( x[i]%2 <=1 ):
        f0[i] = 1
    else:
        f0[i] = -1

#三角波係數
'''
b[1]=0.64
b[2]=0.32
b[3]=0.21
b[4]=0.16
b[5]=0.13
b[6]=0.11
b[7]=0.09
for i in range(0,x.shape[0]): 
    f0[i] = (1 - x[i]%2)
'''


fsum = a[0]/2 



for i in range(1,NWaves): 
    f[i]=a[i]*np.cos(i*2*np.pi*x/L0)+b[i]*np.sin(i*2*np.pi*x/L0)
    fsum=fsum+f[i]
    

colors = plt.cm.jet(np.linspace(0,1,NWaves+1))


Lines = [None] * NWaves

for i in range(1,NWaves): 
    Lines[i] = ax.plot(x, f[i], linewidth=3, color=colors[i])
    
Linef0 = ax.plot(x, f0, linewidth=7, color='red')
Linefsum = ax.plot(x, fsum, linewidth=7, color='black')


plt.show()
