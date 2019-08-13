#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt


r = 0.5
x = np.zeros(31)
x[0] = 0.5
for i in range(30):
    x[i+1] = r*x[i]*(1-x[i])
    
plt.plot(range(31),x)    
plt.xlabel('Iteration')
plt.ylabel('State')
plt.title('Evolution for r=0.5 and x0=0.5')
plt.savefig('figure1.pdf')
plt.close()

    
r0 = 0.5
x0 = np.zeros(31)
x0[0] = 0.5
for i in range(30):
    x0[i+1] = r0*x0[i]*(1-x0[i])

r1 = 0.25
x1 = np.zeros(31)
x1[0] = 0.5
for i in range(30):
    x1[i+1] = r1*x1[i]*(1-x1[i])

r2 = 0.9
x2 = np.zeros(31)
x2[0] = 0.5
for i in range(30):
    x2[i+1] = r2*x2[i]*(1-x2[i])

plt.plot(range(31),x0,label='r=0.5')
plt.plot(range(31),x1,label='r=0.25')
plt.plot(range(31),x2,label='r=0.9')
plt.legend();
plt.xlabel('Iteration')
plt.ylabel('State')
plt.title('Evolution for r=0.5 r=0.25 r = 0.9 when x0=0.5')
plt.savefig('figure2.pdf')
plt.close()


r3 = 1.5
x3 = np.zeros(31)
x3[0] = 0.5
for i in range(30):
    x3[i+1] = r3*x3[i]*(1-x3[i])

plt.plot(range(31),x3)
plt.xlabel('Iteration')
plt.ylabel('State')
plt.title('Evolution for r=1.5 and x0=0.5')
plt.savefig('figure3.pdf')
plt.close()

r4 = 2.5
x4 = np.zeros(31)
x4[0] = 0.5
for i in range(30):
    x4[i+1] = r4*x4[i]*(1-x4[i])

plt.plot(range(31),x4)
plt.xlabel('Iteration')
plt.ylabel('State')
plt.title('Evolution for r=2.5 and x0=0.5')
plt.savefig('figure4.pdf')
plt.close()

r5 = 3.25
x5 = np.zeros(31)
x5[0] = 0.5
for i in range(30):
    x5[i+1] = r5*x5[i]*(1-x5[i])

plt.plot(range(31),x5)
plt.xlabel('Iteration')
plt.ylabel('State')
plt.title('Evolution for r=3.25 and x0=0.5')
plt.savefig('figure5.pdf')
plt.close()

r6 = 3.75
x6 = np.zeros(31)
x6[0] = 0.5
for i in range(30):
    x6[i+1] = r6*x6[i]*(1-x6[i])

plt.plot(range(31),x6)
plt.xlabel('Iteration')
plt.ylabel('State')
plt.title('Evolution for r=3.75 and x0=0.5')
plt.savefig('figure6.pdf')
plt.close()



N=1000
R=np.linspace(0,4,1000)

for j in range(N):
	r=R[j]
	x = np.zeros(N+1)
	x[0]=0.5
	for i in range(N):
		x[i+1]=r*x[i]*(1-x[i])
	if (x[N]<0.000001) or ((abs(x[N-1]-x[N])/x[N])<0.001):
		R[j]=1
	else:
		R[j]=0

		

plt.plot(np.linspace(0,4,1000),R)
plt.yticks(np.arange(2), ('Does not converge', 'Converges'))
plt.xlabel('r')
plt.ylabel('Converge')
plt.title('Convergence for different r values ')
plt.savefig('figure7.pdf')
plt.close()



N=1000
R=np.linspace(0,4,1000)
func=np.zeros(1000)
fun=np.zeros(1000)

for j in range(1000):
	r=R[j]
	if r==0:
		fun[j]=-float("inf")
	else:
		fun[j]=(r-1)*1.0/r
	x = np.zeros(N+1)
	x[0]=0.5
	
	for i in range(N):
		x[i+1]=r*x[i]*(1-x[i])
		
	if (x[N]<.000001) or ((abs(x[N-1]-x[N])/(x[N]*1.0))<0.001):
		func[j]=x[N]
	else:
		func[j]=-0.25



plt.plot(R,func,label='f(r)')
plt.plot(R,fun,label='(r-1)/r')
plt.ylim(-3, 3)
plt.legend()
plt.title('Converges limit for different r values')
plt.xlabel('r value')
plt.ylabel('The limit')
plt.savefig('figure8.pdf')

		





    
