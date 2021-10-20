#用 Python 學通識物理：波動篇
#範例二，畫一個波（行進波）

#匯入動畫需要的函數
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

#設定參數
  #設波長
L0=2
  #設週期
T0=2

  #設繪圖範圍
figuresize_x=6
figuresize_y=1.5



#設定繪圖視窗
fig = plt.figure()
ax = plt.axis([0,figuresize_x,-figuresize_y,figuresize_y])
plt.xlabel('x')
plt.ylabel('y')



x = np.arange(0, figuresize_x+0.05, 0.05)
y = np.sin(2*np.pi*x/L0)
line, = plt.plot(x, y)

def animate(i):
    t=i/20
    #向左前進的波
    #y=np.sin(2*np.pi*x/L0 + 2*np.pi*t/T0)
    #向右前進的波
    y=np.sin(2*np.pi*x/L0 - 2*np.pi*t/T0)
    
    line.set_ydata(y)
    
    return line,


ani = animation.FuncAnimation(fig,
                              animate,
                              frames=1000,
                              interval=20,
                              repeat=True)
plt.show()
