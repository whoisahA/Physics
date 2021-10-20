# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:28:32 2021

@author: aha
"""

# 最小平方法

#設數列 x y
x=[1.0,2.0,3.0,4.0,5.0]
y=[1.4,1.2,3.6,3.2,4.6]

# 數列個數
N=5

# 設總和初值為0
x_sum=0
y_sum=0
x2_sum=0
xy_sum=0

# 計算各總和
for i in range(0,N):
    x_sum=x[i]+x_sum
    y_sum=y[i]+y_sum
    x2_sum=x[i]*x[i]+x2_sum
    xy_sum=x[i]*y[i]+xy_sum

#計算平均值
x_avg=x_sum/N
y_avg=y_sum/N
x2_avg=x2_sum/N
xy_avg=xy_sum/N

#計算 a b
a= (xy_avg-x_avg*y_avg)/(x2_avg-x_avg*x_avg)
b= y_avg - a* x_avg

# 輸出
print('a= %6.2f' % a)
print('b= %6.2f' % b)

# for 使用範例：文字金字塔
for i in range(1,11):
    for j in range(1,12-i):
        print(' ',end='')
    for j in range(1,i+1):
        #print('*',end='')
        print('* ',end='')
    print('')
print('I am AXXXXXXXXX')

import numpy as np
import matplotlib.pyplot as plt



# Python 範例 --- 繪圖1：繪圖區設定
import numpy as np
import matplotlib.pyplot as plt

x1=-20
x2=20
y1=-10
y2=10
plt.axis([x1,x2,y1,y2])

plt.scatter(0,0,s=1000,color='red')
plt.plot([-7,5],[-6,8],linewidth=3,color='blue')

plt.axis('on')
plt.grid(False)

plt.show()



# Python 範例 --- 繪圖2：xy 函數繪圖
import numpy as np
import matplotlib.pyplot as plt

x1=-20
x2=20
y1=-4
y2=4
plt.axis([x1,x2,y1,y2])
plt.axis('on')
plt.grid(True)

x = np.arange(x1, x2, 0.05)
y = np.sin(x)
yp = np.sin(x) + 2

plt.scatter(x,y,s=100,color='red')
plt.plot(x,yp,linewidth=8,color='blue')

# Python 範例 --- 繪圖3：拋物線（參數式繪圖）
import numpy as np
import matplotlib.pyplot as plt

x1=0
x2=25
y1=0
y2=25
plt.axis([x1,x2,y1,y2])
plt.axis('on')
plt.grid(True)

v0=15
theta=45
g=9.8

t = np.arange(0,10,0.1)
x = v0*np.cos(theta*np.pi/180)*t
y = v0*np.sin(theta*np.pi/180)*t - 0.5*g*t*t
plt.plot(x,y,linewidth=2,color='blue')


# for i in range(17):
#    theta = 5 + 5*i
#    t = np.arange(0,10,0.1)
#    x = v0*np.cos(theta*np.pi/180)*t
#    y = v0*np.sin(theta*np.pi/180)*t - 0.5*g*t*t
#    plt.plot(x,y,linewidth=2,color='blue')

plt.show()


plt.scatter(x,y,s=1,color='red')
plt.plot(x,yp,linewidth=8,color='blue')

plt.show()



# Python 範例 --- 繪圖4：玫瑰線
import numpy as np
import matplotlib.pyplot as plt

x1=-1.5
x2=1.5
y1=-1.5
y2=1.5

n=4.0
d=3.0

plt.axis([x1,x2,y1,y2])
#plt.axis('on')
plt.axis('off')
# plt.grid(True)
plt.grid(False)

t = np.arange(0,100*np.pi,0.1)
r = np.cos(n/d*t)
x = r*np.cos(t)
y = r*np.sin(t)

plt.plot(x,y,linewidth=2,color='red')

plt.show()



# Python 範例 --- 計算等比級數總和
# Python 範例 --- while 迴圈
sum = 0   # 總和變數
x = 1   # 
while x > 0.001 :
    print('x=',x)
    sum = sum+x
    x=x/2
    
print('總和=',sum)



# Python 範例 --- 水平拋射
# Python 範例 --- Binary Search

H=10 #起始高度
v0x=10 #水平速度
g=9.8 #重力加速度


t0=0.6 
t1=1.8

y0= H -0.5*g*t0*t0
y1= H -0.5*g*t1*t1

print('y0(%6.3f)= %7.3f ' %(t0, y0))
print('y1(%6.3f)= %7.3f ' %(t1, y1))
print('')

if( y0*y1 < 0 ):
    while( t1-t0 > 0.01 ):
        t2=(t0+t1)/2
        y2= H -0.5*g*t2*t2
        if( y0*y2 > 0 ):
            t0=t2
            y0=y2
        else:
            t1=t2
            y1=y2
        print('y0(%6.3f)= %7.3f ' %(t0, y0))
        print('y1(%6.3f)= %7.3f ' %(t1, y1))
        print('dt = %6.4f ' %(t1-t0))
        print('')
        
        
    t=(t1+t0)/2
    x=v0x*t
    print('飛行時間= %4.2f, 飛行距離= %4.2f' %(t,x ))
else:
    print('y0 與 y1 同號，不一定有解')
    
    

# 碰撞篇 第二週 
# 範例1:平移一顆球

dt=0.2
t_min=0
t_max=10

x=-1
y=-1
vx=0.2
vy=0.1

t=t_min
while t <= t_max :
    t=t+dt
    x=x+vx*dt
    y=y+vy*dt
    print('%5.2f %5.2f %5.2f' %(t,x,y))



# 碰撞篇 第二週 
# 範例2:盒中一顆球

dt=0.2
t_min=0
t_max=40

x=-1
y=-1
vx=-0.25
vy=0.15

t=t_min
while t <= t_max :
    t=t+dt
    x=x+vx*dt
    y=y+vy*dt
    
    if( x > 2 ):
        vx=-abs(vx)
    elif( x < -2):
        vx=abs(vx)
    
    if( y > 2 ):
        vy=-abs(vy)
    elif( y < -2):
        vy=abs(vy)
    
    print('%5.2f %5.2f %5.2f' %(t,x,y)) 
    


# 碰撞篇 第二週 
# 範例3:平移一顆球（動畫）
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


dt=0.2
t_min=0
t_max=10

x=-1
y=-1
vx=0.2
vy=0.16


position=[x,y]
velocity=[vx,vy]

fig = plt.figure()

ax = plt.axis([-2,2,-2,2])

r=1
redDot, = plt.plot(position[0], position[1], 'ro', ms=30)



def animate(t,position,velocity):
    
    position[0]=position[0]+velocity[0]*dt
    position[1]=position[1]+velocity[1]*dt
    
       
    redDot.set_data(position[0], position[1])
    
    
    return

myAnimation = animation.FuncAnimation(fig, animate, 
                        frames=np.arange(t_min, t_max, dt), 
                        fargs=[position,velocity], 
                        interval=10, repeat=False)


plt.show()



# 碰撞篇 第二週 
# 範例4:盒中一顆球（動畫）

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


dt=0.4
t_min=0
t_max=10

x=0
y=0
vx=0.44
vy=-0.17


position=[x,y]
velocity=[vx,vy]

fig = plt.figure()

ax = plt.axis([-2,2,-2,2])

r=1
redDot, = plt.plot(position[0], position[1], 'ro', ms=30)



def animate(t,position,velocity):
    
    position[0]=position[0]+velocity[0]*dt
    position[1]=position[1]+velocity[1]*dt
    
    if position[0] > 2 :
        velocity[0] = - np.abs(velocity[0])
    elif position[0] < -2 :
        velocity[0] =  np.abs(velocity[0])
        
    if position[1] > 2 :
        velocity[1] = - np.abs(velocity[1])
    elif position[1] < -2 :
        velocity[1] =  np.abs(velocity[1])
    
    redDot.set_data(position[0], position[1])
    
    
    return

myAnimation = animation.FuncAnimation(fig, animate, 
                            frames=np.arange(t_min, t_max, dt), 
                            fargs=[position,velocity],
                            interval=10, repeat=True)

plt.show()



#Python 學物理 
#兩個粒子的碰撞
#範例一：碰撞（文字版）

import math

#設定兩個球的起始位置 x0,y0,x1,y1
x0=1
y0=0
x1=-1
y1=0

#設定兩個球的起始速度 v0x,v0y,v1x,v1y
v0x=-1
v0y=0
v1x=1
v1y=0

#設定每一小步代表的時間，及其上下限
dt=0.05
t_min=0
t_max=10
DR01=0.4

print('%5.2f %5.2f %5.2f %5.2f %5.2f' 
          %(t_min,x0,y0,x1,y1))

t=t_min
while t <= t_max :

    #用尤拉法計算新的位置
    t=t+dt
    x0=x0+v0x*dt
    y0=y0+v0y*dt
    x1=x1+v1x*dt
    y1=y1+v1y*dt
    
    #計算等會會用到的量
    dx01 = x0-x1   #兩球x方向的差距
    dy01 = y0-y1   #兩球y方向的差距
    #R01 = 兩球距離   
    R01 =  math.sqrt(dx01*dx01 + dy01*dy01)     
    #用D01判斷兩球位置差距與速度差距是否同向
    # D01 > 0 兩球分開
    # D01 < 0 兩球靠近    
    D01 = dx01*(v0x-v1x) + dy01*(v0y-v1y)

    # 若滿足兩球碰撞條件，則計算新速度
    # 執行距離小於 DR01 且 兩球靠近
    if R01 < DR01 and D01 < 0:
        # 計算質心速度
        vcmx = ( v0x + v1x )/2
        vcmy = ( v0y + v1y )/2
        
        # 計算兩球碰撞前，相對於質心的速度
        v0x_cm = v0x - vcmx
        v0y_cm = v0y - vcmy
        v1x_cm = v1x - vcmx
        v1y_cm = v1y - vcmy

        # 計算兩球碰撞後，相對於質心的速度
        v0xp_cm = (-v0x_cm*(dx01*dx01-dy01*dy01) 
                        - 2* v0y_cm*dx01*dy01)/(R01*R01) 
        v0yp_cm = (-v0y_cm*(dy01*dy01-dx01*dx01) 
                        - 2* v0x_cm*dx01*dy01)/(R01*R01) 
        v1xp_cm = (-v1x_cm*(dx01*dx01-dy01*dy01) 
                        - 2* v1y_cm*dx01*dy01)/(R01*R01) 
        v1yp_cm = (-v1y_cm*(dy01*dy01-dx01*dx01) 
                        - 2* v1x_cm*dx01*dy01)/(R01*R01) 
        
        #由相對於質心的速度，轉換為相對於桌面的速度
        v0x=v0xp_cm + vcmx
        v0y=v0yp_cm + vcmy
        v1x=v1xp_cm + vcmx
        v1y=v1yp_cm + vcmy
    
    # 若球超過邊界，計算反彈速度
    # 檢查第一個球
    if x0 > 2 :
        v0x = - abs(v0x)
    elif x0 < -2 :
        v0x =  abs(v0x)        
    if y0 > 2 :
        v0y = - abs(v0y)
    elif y0 < -2 :
        v0y =  abs(v0y)
    # 檢查第二個球    
    if x1 > 2 :
        v1x = - abs(v1x)
    elif x1 < -2 :
        v1x =  abs(v1x)        
    if y1 > 2 :
        v1y = - abs(v1y)
    elif y1 < -2 :
        v1y = abs(v1y)

    print('%5.2f %5.2f %5.2f %5.2f %5.2f' 
          %(t,x0,y0,x1,y1))



#Python 學物理 
#兩個粒子的碰撞
#範例二：碰撞

#匯入動畫需要的函數庫
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#設定兩個球的起始位置 x0,y0,x1,y1
x0=0
y0=1
x1=0
y1=-1

#設定兩個球的起始速度 v0x,v0y,v1x,v1y
v0x=3
v0y=2
v1x=-1
v1y=-1.5

#設定每一小步代表的時間，及其上下限
dt=0.05
t_min=0
t_max=10

#設定畫面中球的大小及碰撞的距離
DotSize=30
DR01 = 0.35


#設定繪圖視窗
fig = plt.figure()
#設定繪圖視窗的座標範圍
ax = plt.axis([-2.2,2.2,-2.2,2.2])

# 將位置與速度轉成動畫需要的數列格式
position=np.empty([2,2])
velocity=np.empty([2,2])
position[0][0]=x0
position[0][1]=y0
position[1][0]=x1
position[1][1]=y1
velocity[0][0]=v0x
velocity[0][1]=v0y
velocity[1][0]=v1x
velocity[1][1]=v1y

#先在視窗中，準備兩個球（點）
#(1) 先為兩個點準備記憶體
Dots = [None] * 2
#(2) 畫兩個點
Dots[0], = plt.plot(position[0][0], position[0][1], 'ro', ms=DotSize,color='red')
Dots[1], = plt.plot(position[1][0], position[1][1], 'ro', ms=DotSize,color='green')


#定義動畫中，重複執行的動作
def animate(t,position,velocity):
    
    # 為了方便學習
    # 還是把位置和速度用我們比較孰悉的方式表示
    x0=position[0][0]
    y0=position[0][1]
    x1=position[1][0]
    y1=position[1][1]
    v0x=velocity[0][0]
    v0y=velocity[0][1]
    v1x=velocity[1][0]
    v1y=velocity[1][1]
    
    #用尤拉法計算新的位置
    x0=x0+v0x*dt
    y0=y0+v0y*dt
    x1=x1+v1x*dt
    y1=y1+v1y*dt
    
    #計算等會會用到的量
    dx01 = x0-x1   #兩球x方向的差距
    dy01 = y0-y1   #兩球y方向的差距
    R01 =  np.sqrt(dx01*dx01 + dy01*dy01) #兩球距離
    
    #用D01判斷兩球位置差距與速度差距是否同向
    # D01 > 0 兩球分開
    # D01 < 0 兩球靠近    
    D01 = dx01*(v0x-v1x) + dy01*(v0y-v1y)

    # 若滿足兩球碰撞條件，則計算新速度
    # 執行距離小於 DR01 且 兩球靠近
    if R01 < DR01 and D01 < 0:
        # 計算質心速度
        vcmx = ( v0x + v1x )/2
        vcmy = ( v0y + v1y )/2
        
        # 計算兩球碰撞前，相對於質心的速度
        v0x_cm = v0x - vcmx
        v0y_cm = v0y - vcmy
        v1x_cm = v1x - vcmx
        v1y_cm = v1y - vcmy

        # 計算兩球碰撞後，相對於質心的速度
        v0xp_cm = (-v0x_cm*(dx01*dx01-dy01*dy01) 
                        - 2* v0y_cm*dx01*dy01)/(R01*R01) 
        v0yp_cm = (-v0y_cm*(dy01*dy01-dx01*dx01) 
                        - 2* v0x_cm*dx01*dy01)/(R01*R01) 
        v1xp_cm = (-v1x_cm*(dx01*dx01-dy01*dy01) 
                        - 2* v1y_cm*dx01*dy01)/(R01*R01) 
        v1yp_cm = (-v1y_cm*(dy01*dy01-dx01*dx01) 
                        - 2* v1x_cm*dx01*dy01)/(R01*R01) 
        
        #由相對於質心的速度，轉換為相對於桌面的速度
        v0x=v0xp_cm + vcmx
        v0y=v0yp_cm + vcmy
        v1x=v1xp_cm + vcmx
        v1y=v1yp_cm + vcmy
    
    # 若球超過邊界，計算反彈速度
    # 檢查第一個球
    if x0 > 2 :
        v0x = - np.abs(v0x)
    elif x0 < -2 :
        v0x =  np.abs(v0x)        
    if y0 > 2 :
        v0y = - np.abs(v0y)
    elif y0 < -2 :
        v0y =  np.abs(v0y)
    # 檢查第二個球    
    if x1 > 2 :
        v1x = - np.abs(v1x)
    elif x1 < -2 :
        v1x =  np.abs(v1x)        
    if y1 > 2 :
        v1y = - np.abs(v1y)
    elif y1 < -2 :
        v1y = np.abs(v1y)

    # 將位置與速度轉成動畫需要的格式
    position[0][0]=x0
    position[0][1]=y0
    position[1][0]=x1
    position[1][1]=y1
    velocity[0][0]=v0x
    velocity[0][1]=v0y
    velocity[1][0]=v1x
    velocity[1][1]=v1y
    
    #更新球的圖形資料       
    Dots[0].set_data(position[0][0], position[0][1])
    Dots[1].set_data(position[1][0], position[1][1])
    
    return #結束動畫中，重複執行的動作

#呼叫動畫執行命令
myAnimation = animation.FuncAnimation(fig, animate, \
                                      frames=np.arange(t_min, t_max, dt), \
                                      fargs=[position,velocity],interval=10, repeat=True)

#把圖秀出來
plt.show()



#Python 學物理 
#盒中的一堆球
#範例一：盒中的一堆球

#匯入動畫需要的函數
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#設定每一小步代表的時間，及其上下限
dt=0.05
t_min=0
t_max=10

#設定畫面中球的數量、大小及碰撞的距離
NDots=15
DotSize=20
DRij = 0.2


#設定繪圖視窗
fig = plt.figure()
#設定繪圖視窗的座標範圍
ax = plt.axis([-2.1,2.1,-2.1,2.1])


#設定球的初始位置與速度
#(1) 先為各球的位置與速度準備記憶體
position=np.empty([NDots,2])
velocity=np.empty([NDots,2])

#(2) 隨機設定各球的位置與速度
T1=8
v0=0.5*np.sqrt(T1)
for i in range(0,NDots):
    position[i][0]=4*np.random.rand()-2
    position[i][1]=4*np.random.rand()-2
    theta=2 * np.pi * np.random.rand()
    velocity[i][0]=v0*np.cos(theta)
    velocity[i][1]=v0*np.sin(theta)
    
#(3) 先畫 NDots 個點
Dots = [None] * NDots
for i in range(NDots):
    Dots[i], = plt.plot(position[i][0], position[i][1], 'ro', ms=DotSize)


#定義動畫中，重複執行的動作
def animate(t,position,velocity):
        
    #計算每個球的新位置與速度
    for i in range(NDots):
        #用尤拉法計算新的位置
        position[i][0]=position[i][0]+velocity[i][0]*dt
        position[i][1]=position[i][1]+velocity[i][1]*dt

        #計算粒子對
        for j in range(i+1,NDots,1):
            #計算等會會用到的量
              #兩球x方向的差距
            dxij = position[i][0]-position[j][0]
              #兩球y方向的差距
            dyij = position[i][1]-position[j][1]
              #兩球距離
            Rij = np.sqrt( dxij*dxij + dyij*dyij )
              #用Dij判斷兩球位置差距與速度差距是否同向
                # Dij > 0 兩球分開
                # Dij < 0 兩球靠近   
            Dij = dxij*(velocity[i][0]-velocity[j][0]) + dyij*(velocity[i][1]-velocity[j][1])
            
            # 若滿足兩球碰撞條件，則計算新速度
            # 執行距離小於 DRij 且 兩球靠近
            if Rij < DRij and Dij < 0:
                # 計算質心速度
                vcmx = ( velocity[i][0] + velocity[j][0] )/2
                vcmy = ( velocity[i][1] + velocity[j][1] )/2
                
                # 計算兩球碰撞前，相對於質心的速度
                vix_cm = velocity[i][0] - vcmx
                viy_cm = velocity[i][1] - vcmy
                vjx_cm = velocity[j][0] - vcmx
                vjy_cm = velocity[j][1] - vcmy
                
                # 計算兩球碰撞後，相對於質心的速度
                vixp_cm = (-vix_cm*(dxij*dxij-dyij*dyij) 
                        - 2* viy_cm*dxij*dyij)/(Rij*Rij) 
                viyp_cm = (-viy_cm*(dyij*dyij-dxij*dxij) 
                        - 2* vix_cm*dxij*dyij)/(Rij*Rij) 
                vjxp_cm = (-vjx_cm*(dxij*dxij-dyij*dyij) 
                        - 2* vjy_cm*dxij*dyij)/(Rij*Rij) 
                vjyp_cm = (-vjy_cm*(dyij*dyij-dxij*dxij) 
                        - 2* vjx_cm*dxij*dyij)/(Rij*Rij) 

                #由相對於質心的速度，轉換為相對於桌面的速度
                velocity[i][0] = vixp_cm + vcmx
                velocity[i][1] = viyp_cm + vcmy
                velocity[j][0] = vjxp_cm + vcmx
                velocity[j][1] = vjyp_cm + vcmy


        # 若球超過邊界，計算反彈速度     
        if position[i][0] > 2 :
            velocity[i][0] = - np.abs(velocity[i][0])
        elif position[i][0] < -2 :
            velocity[i][0] =  np.abs(velocity[i][0])
        
        if position[i][1] > 2 :
            velocity[i][1] = - np.abs(velocity[i][1])
        elif position[i][1] < -2 :
            velocity[i][1] =  np.abs(velocity[i][1])
    
        #更新球的圖形資料
        Dots[i].set_data(position[i][0], position[i][1])
        
    return #結束動畫中，重複執行的動作

#呼叫動畫執行命令
myAnimation = animation.FuncAnimation(fig, animate, \
                                      frames=np.arange(t_min, t_max, dt), \
                                      fargs=[position,velocity],interval=5, repeat=True)
#把圖秀出來
plt.show()



#Python 學物理 
#盒中的一堆球
#範例二：驗證PV=NkT

#匯入動畫需要的函數
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#設定每一小步代表的時間，及其上下限
dt=0.05
t_min=0
t_max=100
t=t_min

#設定畫面中球的數量、大小及碰撞的距離
NDots=30
DotSize=20
DRij = 0.2


#設定繪圖視窗
fig = plt.figure()
#設定繪圖視窗的座標範圍
ax = plt.axis([-2.1,2.1,-2.1,2.1])

impulse=np.zeros([1,2])
T1=np.zeros([1])

#設定球的初始位置與速度
#(1) 先為各球的位置與速度準備記憶體
position=np.empty([NDots,2])
velocity=np.empty([NDots,2])

#(2) 隨機設定各球的位置與速度
T1[0]=10
v0=0.5*np.sqrt(T1[0])
for i in range(0,NDots):
    position[i][0]=4*np.random.rand()-2
    position[i][1]=4*np.random.rand()-2
    theta=2 * np.pi * np.random.rand()
    velocity[i][0]=v0*np.cos(theta)
    velocity[i][1]=v0*np.sin(theta)
    
#(3) 先畫 NDots 個點
Dots = [None] * NDots
for i in range(NDots):
    Dots[i], = plt.plot(position[i][0], position[i][1], 'ro', ms=DotSize)



#定義動畫中，重複執行的動作
def animate(t,position,velocity):
        
    #計算每個球的新位置與速度
    for i in range(NDots):
        #用尤拉法計算新的位置
        position[i][0]=position[i][0]+velocity[i][0]*dt
        position[i][1]=position[i][1]+velocity[i][1]*dt

        #計算粒子對
        for j in range(i+1,NDots,1):
            #計算等會會用到的量
              #兩球x方向的差距
            dxij = position[i][0]-position[j][0]
              #兩球y方向的差距
            dyij = position[i][1]-position[j][1]
              #兩球距離
            Rij = np.sqrt( dxij*dxij + dyij*dyij )
              #用Dij判斷兩球位置差距與速度差距是否同向
                # Dij > 0 兩球分開
                # Dij < 0 兩球靠近   
            Dij = dxij*(velocity[i][0]-velocity[j][0]) + dyij*(velocity[i][1]-velocity[j][1])
            
            # 若滿足兩球碰撞條件，則計算新速度
            # 執行距離小於 DRij 且 兩球靠近
            if Rij < DRij and Dij < 0:
                # 計算質心速度
                vcmx = ( velocity[i][0] + velocity[j][0] )/2
                vcmy = ( velocity[i][1] + velocity[j][1] )/2
                
                # 計算兩球碰撞前，相對於質心的速度
                vix_cm = velocity[i][0] - vcmx
                viy_cm = velocity[i][1] - vcmy
                vjx_cm = velocity[j][0] - vcmx
                vjy_cm = velocity[j][1] - vcmy
                
                # 計算兩球碰撞後，相對於質心的速度
                vixp_cm = (-vix_cm*(dxij*dxij-dyij*dyij) 
                        - 2* viy_cm*dxij*dyij)/(Rij*Rij) 
                viyp_cm = (-viy_cm*(dyij*dyij-dxij*dxij) 
                        - 2* vix_cm*dxij*dyij)/(Rij*Rij) 
                vjxp_cm = (-vjx_cm*(dxij*dxij-dyij*dyij) 
                        - 2* vjy_cm*dxij*dyij)/(Rij*Rij) 
                vjyp_cm = (-vjy_cm*(dyij*dyij-dxij*dxij) 
                        - 2* vjx_cm*dxij*dyij)/(Rij*Rij) 

                #由相對於質心的速度，轉換為相對於桌面的速度
                velocity[i][0] = vixp_cm + vcmx
                velocity[i][1] = viyp_cm + vcmy
                velocity[j][0] = vjxp_cm + vcmx
                velocity[j][1] = vjyp_cm + vcmy
                                
        # 若球超過邊界，計算反彈速度     
        if position[i][0] > 2 :
            velocity[i][0] = - np.abs(velocity[i][0])
            impulse[0][0]=impulse[0][0]+1
            impulse[0][1]=impulse[0][1]+2*np.abs(velocity[i][0])
            # print("次數= %d 總衝量= %7.2f"  % (impulse[0][0],impulse[0][1]) )
            print("%3.0f %5d %7.2f"  % (T1[0], impulse[0][0],impulse[0][1]) )
        elif position[i][0] < -2 :
            velocity[i][0] =  np.abs(velocity[i][0])
            impulse[0][0]=impulse[0][0]+1
            impulse[0][1]=impulse[0][1]+2*np.abs(velocity[i][0])
            #print("次數= %d 總衝量= %7.2f"  % (impulse[0][0],impulse[0][1]) )
            print("%3.0f %5d %7.2f"  % (T1[0], impulse[0][0],impulse[0][1]) )
        
        if position[i][1] > 2 :
            velocity[i][1] = - np.abs(velocity[i][1])
            impulse[0][0]=impulse[0][0]+1
            impulse[0][1]=impulse[0][1]+2*np.abs(velocity[i][1])
            #print("次數= %d 總衝量= %7.2f"  % (impulse[0][0],impulse[0][1]) )
            print("%3.0f %5d %7.2f"  % (T1[0], impulse[0][0],impulse[0][1]) )
        elif position[i][1] < -2 :
            velocity[i][1] =  np.abs(velocity[i][1])
            impulse[0][0]=impulse[0][0]+1
            impulse[0][1]=impulse[0][1]+2*np.abs(velocity[i][1])
            #print("次數= %d 總衝量= %7.2f"  % (impulse[0][0],impulse[0][1]) )
            print("%3.0f %5d %7.2f"  % (T1[0], impulse[0][0],impulse[0][1]) )
    
        #更新球的圖形資料
        Dots[i].set_data(position[i][0], position[i][1])
        
        '''t=t+dt
        if(t >= t_max):
            print("t = %7.2f %d" %(t,n))
            n=n+1
            t=t_min'''
        #print("撞牆次數= %d" %impulse[0][0] )
    return #結束動畫中，重複執行的動作

#呼叫動畫執行命令
myAnimation = animation.FuncAnimation(fig, animate, \
                                      frames=np.arange(t_min, t_max, dt), \
                                      fargs=[position,velocity],interval=5, repeat=False)
#把圖秀出來
plt.show()

print("撞牆次數= %d" %impulse[0][0] )




