# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 14:06:22 2021

@author: aha
"""

#用 Python 學通識物理：波動篇
#第四週 範例一，送你一道彩虹

import numpy as np
import matplotlib.pyplot as plt

x1=-10
x2=10
y1=0
y2=15
ax = plt.axis([x1,x2,y1,y2])

plt.axis('on')
plt.grid(False)

theta = np.linspace(0, np.pi, 100 )
x=np.cos(theta)
y=np.sin(theta)

# 第一種表示法
'''
plt.plot(11*x,11*y,linewidth=18,color='magenta')
plt.plot(10*x,10*y,linewidth=18,color='cyan')
plt.plot(9*x,9*y,linewidth=18,color='blue')
plt.plot(8*x,8*y,linewidth=18,color='green')
plt.plot(7*x,7*y,linewidth=18,color='yellow')
plt.plot(6*x,6*y,linewidth=18,color='orange')
plt.plot(5*x,5*y,linewidth=18,color='red')
'''

# 第二種表示法
'''
n=9
colors = plt.cm.jet(np.linspace(0,1,n))

plt.plot(11*x,11*y,linewidth=18,color=colors[1])
plt.plot(10*x,10*y,linewidth=18,color=colors[2])
plt.plot(9*x,9*y,linewidth=18,color=colors[3])
plt.plot(8*x,8*y,linewidth=18,color=colors[4])
plt.plot(7*x,7*y,linewidth=18,color=colors[5])
plt.plot(6*x,6*y,linewidth=18,color=colors[6])
plt.plot(5*x,5*y,linewidth=18,color=colors[7])
'''

# 第三種表示法

n=9
colors = plt.cm.jet(np.linspace(0,1,n))

for i in range(5,12):
    plt.plot(i*x,i*y,linewidth=18,color=colors[12-i])


plt.show()