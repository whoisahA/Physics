# Python 範例 --- 波動篇第三週
#範例二，兩個水波的干涉

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
plt.axis('off')

A = np.zeros((100,100))

for i in range(100):
    for j in range(100):
        x= -10+20*j/100
        y= -10+20*i/100
        r1=np.sqrt((x-6)*(x-6)+y*y)  
        r2=np.sqrt((x+6)*(x+6)+y*y)
        A[i][j]=np.sin( np.pi*2*(r1/5))+np.sin( np.pi*2*(r2/5))

wave = plt.imshow(A, cmap=plt.get_cmap('winter'))



def animate(t):
    for i in range(100):
        for j in range(100):
            x= -20+40*j/100
            y= -20+40*i/100
            r1=np.sqrt((x-6)*(x-6)+y*y)  
            r2=np.sqrt((x+6)*(x+6)+y*y)
            A[i][j]=np.sin( np.pi*2*(r1/5 - 0.2*t))+np.sin( np.pi*2*(r2/5 - 0.2*t))
    wave.set_array(A)
    return wave


ani = animation.FuncAnimation(fig,
                              animate,
                              frames=1000,
                              interval=20,
                              repeat=True)



# plt.colorbar()
plt.show()  