import numpy as np 


def rosebrock(x):
	x= np.copy(x)
	
	return -1*sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rastrigin(x):
	x = np.copy(x)
	x -= 10.0
	if not np.isscalar(x[0]):
		N = len(x[0])
		return -np.array([10 * N + sum(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])
	N = len(x)
	return -(10 * N + sum(x**2 - 10 * np.cos(2 * np.pi * x)))




def dejong(x): 
	x = np.copy(x)
	if any(x<-1) or any(x>1):
		return -1*10*sum(x**2)

	return -1*sum(x**2)



def hyperEllipsoid(x):

	x = np.copy(x)
	N =[i+1 for i in range(len(x))]

	return -1*np.dot(x**2,N)

def schwefel(x): 

	x = np.copy(x)
	x+=430
	if any(x>500) or any(x<-500):
		return -1*10000000
	val=-1*sum(x*np.sin(np.sqrt(abs(x))))
	val+=418.9829*len(x)
	return -1*val 


def griewangk(x):
	x = np.copy(x)
	N = [i+1 for i in range(len(x))]
	N = np.sqrt(N)
	if any(x>600) or any(x<-600):
		return -1*100000
	val = np.cos(x/N)
	if any(abs(val)<0.0001):
		val=0 
	else:
		val=np.exp(sum(np.log(val)))

	
	
	return -1*(1+sum(x**2)/4000-val)