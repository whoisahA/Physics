#用 Python 學通識物理：波動篇
#範例四，拍頻

#匯入動畫需要的函數
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

#設定參數
  #設波長
L0=1
L1=1

  #設週期
  #前進方向：正號+ 向左，負號- 向右
T0=3.1
T1=3


  #設繪圖範圍
figuresize_x=6
figuresize_y=4



#設定繪圖視窗
fig = plt.figure()
ax = plt.axis([0,figuresize_x,-figuresize_y,figuresize_y])
plt.xlabel('x')
plt.ylabel('y')



x = np.arange(0, figuresize_x+0.02, 0.02)
y0 = np.sin(2*np.pi*x/L0)
line0, = plt.plot(x, y0,color='red')

y1 = np.sin(2*np.pi*x/L1)
line1, = plt.plot(x, y1,color='green')

y2 = y0+y1
line2, = plt.plot(x, y2,linewidth=8,color='blue')



def animate(i):
    t=i/10
    #向左前進的波
    #y=np.sin(2*np.pi*x/L0 + 2*np.pi*t/T0)
    #向右前進的波
    y0=np.sin(2*np.pi*x/L0 + 2*np.pi*t/T0)
    y1=np.sin(2*np.pi*x/L1 + 2*np.pi*t/T1)
    y2 = y0+y1
    
    line0.set_ydata(y0)
    line1.set_ydata(y1)
    line2.set_ydata(y2)
    
    
    return


ani = animation.FuncAnimation(fig,
                              animate,
                              frames=1000,
                              interval=20,
                              repeat=True)
plt.show()
