import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import pickle 
import math 
from testbed import *
import os
os.system('ffmpeg') 

fitfunc=rosebrock
best_x, best_y=1,1 


def readData(name):
	pickle_in=open(name+".pickle","rb")
	data=pickle.load(pickle_in)
	return data 


# Create new Figure and an Axes which fills it.
method="PEPGVRosebrock"
data=readData(method)
NPOPULATION=len(data[0])
MAXITER=len(data)
# print(NPOPULATION)


fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
# ax.set_xlim(-20, 20), ax.set_xticks([])
# ax.set_ylim(-20, 20), ax.set_yticks([])



objects=np.zeros(NPOPULATION, dtype=[('position', float, 2),
	                          ('size',     float, 1),
	                          ('growth',   float, 1),
                               ])


objects['position']=data[0] # done 
objects['growth']=np.zeros(NPOPULATION)
objects['size']=np.ones(NPOPULATION)*5
print(objects['size'])
print(len(data))


x_min,x_max,y_min,y_max=np.min(data[:,:,0]), np.max(data[::,0]), np.min(data[:,:,1]), np.max(data[:,:,1])
x_min,y_min=math.floor(x_min), math.floor(y_min)
x_max,y_max=math.ceil(x_max),math.ceil(y_max)
# hold 

ax.set_xlim(x_min,x_max)
ax.set_ylim(y_min,y_max)


x=np.arange(x_min,x_max,0.01)
y=np.arange(y_min,y_max,0.01)
X,Y= np.meshgrid(x,y)
Z= np.zeros(X.shape)
h,w=X.shape 
for i in range(h):
	for j in range(w):
		Z[i,j]=fitfunc([X[i,j],Y[i,j]])
print(Z.shape)
levels=[-200,-100,-10,-4,-2,-1,-0.5,0,1,0.5,2,4,10,100,200]
CS=plt.contour(X,Y,Z,levels)
plt.clabel(CS,inline=1,fontsize=9)



plt.scatter([best_x],[best_y],s=30,facecolor='red')



scat = ax.scatter(objects['position'][:,0], objects['position'][:,1],lw=0.5,s=objects['size'],facecolor='black')



def update(frame_number):
	#pass 
	current_index = frame_number % MAXITER

	objects['position']= data[current_index]
	# print(objects['position'])
	objects['size']=np.ones(NPOPULATION)*5


	scat.set_sizes(objects['size'])
	scat.set_offsets(objects['position'])
	# scat.set_edgecolors('blue')








# # Construct the animation, using the update function as the animation
# # director.
ani = FuncAnimation(fig, update,frames=MAXITER, interval=10)
# # # Writer = animation.writers['ffmpeg']
# # # mywriter = Writer(fps=15, bitrate=1800)
# # # FFwriter = animation.FFMpegWriter()

ani.save("./"+method+".mp4", fps=10)
# plt.show()