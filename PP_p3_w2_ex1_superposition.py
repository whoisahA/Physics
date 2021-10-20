#用 Python 學通識物理：波動篇
#範例一，兩個波的疊加

#匯入動畫需要的函數
from matplotlib import pyplot as plt
import numpy as np

#設定參數
  #設波長 最大振幅 起始相位
L0=2
y00=1.2
phi0=0*(2*np.pi)

L1=3.2
y01=1.2
phi1=0.1*(2*np.pi)

#是否合併繪圖 S1=1 合併
S1=0

#設繪圖範圍
figuresize_x=24
figuresize_y=1.5


#設定繪圖視窗
fig = plt.figure()
if (S1 == 0):
    ax = plt.axis([0,figuresize_x,-4,12])
else:
    ax = plt.axis([0,figuresize_x,-3,3])
plt.xlabel('x')
plt.ylabel('y')



x = np.arange(0, figuresize_x+0.02, 0.02)
y0 = y00*np.sin(2*np.pi*x/L0+phi0)
y1 = y01*np.sin(2*np.pi*x/L1+phi1)
y2 = y0+y1

if (S1 == 0):
    line0 = plt.plot(x, y0+10,color='red')
    line1 = plt.plot(x, y1+5,color='green')
    line2 = plt.plot(x, y2,linewidth=8,color='blue')
else:
    line0 = plt.plot(x, y0,color='red')
    line1 = plt.plot(x, y1,color='green')
    line2 = plt.plot(x, y2,linewidth=8,color='blue')




plt.show()
