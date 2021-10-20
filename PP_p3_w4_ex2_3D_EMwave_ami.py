# Python 範例 --- 波動篇第四週
#範例二，電磁波 3D 示範

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


x_start=-5
x_stop=5

A0=2
L0=4
T0=1

fig = plt.figure()

ax = plt.axes(projection='3d')

ax.set_xlim(x_start,x_stop)
ax.set_ylim(x_start,x_stop)
ax.set_zlim(x_start,x_stop)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#設定x軸的數列
x = np.arange(x_start,x_stop, 0.15)

NLines=3
Data = np.empty([NLines,3,x.shape[0]])

#設定波前進軸的量（只有x軸）
Data[0][0]=x
Data[0][1]=0*x
Data[0][2]=0*x

#設定波的電場的量（x-y軸）
Data[1][0]=x
Data[1][1]=A0*np.cos(2*np.pi*x/L0)
Data[1][2]=0*x

#設定波的磁場的量（x-z軸）
Data[2][0]=x
Data[2][1]=0*x
Data[2][2]=A0*np.cos(2*np.pi*x/L0)



Lines = [None] * NLines

#繪出波前進軸
Lines[0], = ax.plot(Data[0][0], Data[0][1], Data[0][2], linewidth=5, color='red')
#繪出波的電場曲線
Lines[1], = ax.plot(Data[1][0], Data[1][1], Data[1][2], linewidth=5, color='blue')
#繪出波的磁場曲線
Lines[2], = ax.plot(Data[2][0], Data[2][1], Data[2][2], linewidth=5, color='green')



def animate(i):
    t=i/20
    
    #更新波的電場量
    Data[1][1]=A0*np.cos(2*np.pi*x/L0 - 2*np.pi*t/T0)
    #更新波的磁場量
    Data[2][2]=A0*np.cos(2*np.pi*x/L0 - 2*np.pi*t/T0)

    for i in range(2):
        Lines[i].set_xdata(Data[i][0])
        Lines[i].set_ydata(Data[i][1])
        Lines[i].set_3d_properties(Data[i][2])

    return Lines,

ani = animation.FuncAnimation(fig,
                              animate,
                              frames=1000,
                              interval=20,
                              repeat=True)


plt.show()
