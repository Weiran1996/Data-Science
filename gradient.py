

#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt





def f(x1,x2):
    return 5*x1**2 + 4*x1*x2 + 3*x2**2





def g(x1,x2,f,t):
    bottom = ((10*x1 + 4*x2)**2 + (6*x2 + 4*x1)**2)**0.5
    m = x1-(t*(10*x1+4*x2)/bottom) 
    n = x2-(t*(6*x2+4*x1)/bottom)
    return f(m,n)





def grad(x1,x2,t,f,g):
    m = g(x1,x2,f,t+10**(-10)) - g(x1,x2,f,t)
    n = 10**(-10)
    return m/n





a = 0.3
b = 0.3
x1 = 5
x2 = 0
x1_list1=[5]
x2_list1=[0]
t = 1
for i in range (10):
    while(g(x1,x2,f,t)>g(x1,x2,f,0) + a*grad(x1,x2,0,f,g)*t):
        t = t*b
    gradient1 = 10*x1 + 4*x2
    gradient2 = 6*x2 + 4*x1
    norm = ((10*x1 + 4*x2)**2 + (6*x2 + 4*x1)**2)**0.5
    x1 = x1 - (t*gradient1/norm)
    x2 = x2 - (t*gradient2/norm)
    x1_list1.append(x1)
    x2_list1.append(x2)
    print t
    print f(x1,x2)
print x1_list1
print x2_list1

delta = 0.0025
x = np.arange(-5.0, 5.0, delta)
y = np.arange(-5.0, 5.0, delta)
X, Y = np.meshgrid(x,y)
Z = 5*X**2 + 4*X*Y + 3*Y**2
plt.figure()
plt.plot(x1_list1,x2_list1,'ro')
CS = plt.contour(X,Y,Z, np.linspace(1,150,10))
plt.xlabel('domain of variable x1')
plt.ylabel('domain of variable x2')
plt.title('Plot 1 : Minimize f with backtracking algorithm')
plt.savefig('figure1.pdf')
plt.close()





x1 = 5
x2 = 0
x1_list2=[5]
x2_list2=[0]
gradd = (grad(x1,x2,0,f,g)-grad(x1,x2,-10**(-3),f,g))/10**(-3)
t = abs(grad(x1,x2,0,f,g)/gradd)
for i in range (5):
    while(g(x1,x2,f,t)>g(x1,x2,f,0) + a*grad(x1,x2,0,f,g)*t):
        gradd = (grad(x1,x2,0,f,g)-grad(x1,x2,-10**(-3),f,g))/10**(-3)
        t = abs(grad(x1,x2,0,f,g)/gradd)
    gradient1 = 10*x1 + 4*x2
    gradient2 = 6*x2 + 4*x1
    norm = ((10*x1 + 4*x2)**2 + (6*x2 + 4*x1)**2)**0.5
    x1 = x1 - (t*gradient1/norm)
    x2 = x2 - (t*gradient2/norm)
    x1_list2.append(x1)
    x2_list2.append(x2)
    print t
    print f(x1,x2)
print x1_list2
print x2_list2


plt.figure()
plt.plot(x1_list2,x2_list2,'ro')
CS = plt.contour(X,Y,Z, np.linspace(1,150,10))
plt.xlabel('domain of variable x1')
plt.ylabel('domain of variable x2')
plt.title('Plot 2 : Minimize f with Newtons method')
plt.savefig('figure2.pdf')
plt.close()




def gx(x):
    return x**4 - 2*x**3 -1.5*x*x + 2.5*x + 1

def gd(x):
    return 4*x**3 - 6*x**2 - 3*x + 2.5

def gt(x,t,gx,gd):
    return gx(x-t*(gd(x)/abs(gd(x))))

def gp(x,t):
              return -4*(x-t)**3 + 6*(x-t)**2 + 3*(x-t) - 2.5





a = 0.3
b = 0.3
xn = 2.9
xlist1 = [2.9]
ylist1 = [gx(2.9)]
t = 1
for i in range (10):
    while gt(xn,t,gx,gd)>(gt(xn,0,gx,gd) + a*gp(xn,0)*t):
        t = t*b
    xn = xn - t*(gd(xn)/abs(gd(xn)))
    print xn
    xlist1.append(xn)
    ylist1.append(gx(xn))
print xlist1
print ylist1
t1 = np.arange(-1.5,3,0.01)
t2 = gx(t1)
plt.plot(t1,t2)
plt.plot(xlist1,ylist1,'ro')
plt.xlabel('domain of variable x')
plt.ylabel('g(x)')
plt.title('Plot 3 : Backtracking when starting point is 2.9')
plt.savefig('figure3.pdf')
plt.close()




xn = -1.5
xlist2 = [-1.5]
ylist2 = [gx(-1.5)]
t = 1
for i in range (10):
    while gt(xn,t,gx,gd)>(gt(xn,0,gx,gd) + a*gp(xn,0)*t):
        t = t*b
    xn = xn - t*(gd(xn)/abs(gd(xn)))
    print xn
    xlist2.append(xn)
    ylist2.append(gx(xn))
print xlist2
print ylist2
t1 = np.arange(-2,2.5,0.01)
t2 = gx(t1)
plt.plot(t1,t2)
plt.plot(xlist2,ylist2,'ro')
plt.xlabel('domain of variable x')
plt.ylabel('g(x)')
plt.title('Plot 4 : Backtracking when starting point is -1.5')
plt.savefig('figure4.pdf')

