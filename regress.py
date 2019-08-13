
# coding: utf-8

# In[110]:


#!/usr/bin/env python


# In[115]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# In[116]:


with open('/classes/ece2720/pe4/small.csv') as sm:
    while 1:
        sma = sm.read(1)
        if not sma:
            break
        print 'read a char: ', sma
        print sma.encode("hex")


# In[119]:


traindata = pd.read_csv('/classes/ece2720/pe4/train.csv')
x = traindata.loc[:, 'Cement (component 1)(kg in a m^3 mixture)' : 'Age (day)'].values
y = traindata.loc[:, 'Concrete compressive strength(MPa'].values

testdata = pd.read_csv('/classes/ece2720/pe4/test.csv')
x1 = testdata.loc[:, 'Cement (component 1)(kg in a m^3 mixture)' : 'Age (day)'].values
y1 = testdata.loc[:, 'Concrete compressive strength(MPa, megapascals) '].values

smalldata = pd.read_csv('/classes/ece2720/pe4/small.csv')
x2 = smalldata.loc[:, 'Cement (component 1)(kg in a m^3 mixture)' : 'Age (day)'].values
y2 = smalldata.loc[:, 'Concrete compressive strength(MPa'].values


# In[120]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(x, y) 


# In[121]:


a = regressor.coef_
print "The regressors of the train data is: ", regressor.coef_
b = regressor.intercept_
print "The intercept of the train data is: ", regressor.intercept_


# In[122]:


regressor2 = LinearRegression()  
regressor2.fit(x2, y2) 


# In[123]:


a2 = regressor2.coef_
print "The regressors of the small data is: ", regressor2.coef_
b2 = regressor2.intercept_
print "The regressors of the small data is: ", regressor2.intercept_
print 'The length of vector a is ', np.linalg.norm(regressor2.coef_)


# In[124]:


def squared_error(x,y,regress,intercept):
    result = 0.0
    cross = 0.0
    for i in range(x.shape[0]):
        for j in range(len(regress)):
            cross = cross + regress[j]*x[i,j]
        result = result + (y[i] - cross - intercept)**2
        cross = 0.0
    return result


# In[125]:


def y_squared(x,y,regress,intercept):
    result = 0.0
    for i in range(x.shape[0]):
        result = result + (y[i] - sum(y)/len(y))**2
    return result


# In[126]:


def R_squared(x,y,regress,intercept):
    return 1 - squared_error(x,y,regress,intercept)/y_squared(x,y,regress,intercept)


# In[127]:


print "The R^2 of the train data is: ", R_squared(x,y,a,b)
print "The test R^2 of the train data is: ", R_squared(x1,y1,a,b)
print "The R^2 of the small data is: ", R_squared(x2,y2,a2,b2)
print "The test R^2 of the small data is: ", R_squared(x1,y1,a2,b2)


# In[128]:


from sklearn.linear_model import Ridge
import numpy as np
r1 = []
r2 = []
lamda1 = []
for i in range(1,101):
    clf = Ridge(alpha=x2.shape[0]*(1.2)**i)
    clf.fit(x2, y2)
    #print clf.coef_
    #print clf.intercept_
    r1.append(R_squared(x2,y2,clf.coef_, clf.intercept_))
    r2.append(R_squared(x1,y1,clf.coef_, clf.intercept_))
    lamda1.append(x2.shape[0]*(1.2)**i)

plt.figure()
plt.plot(lamda1, r1)
plt.plot(lamda1, r2)
plt.xlabel('the value of lamda')
plt.ylabel('R squared')
plt.title('Plot 1 : Function of lamda and R squared')
plt.savefig('figure1.pdf')


# In[129]:


print np.argmax(r2)
print "The best lamda for Rigid Regression is: ", lamda1[np.argmax(r2)]
clf = Ridge(alpha=x2.shape[0]*(1.2)**(np.argmax(r2)+1))
clf.fit(x2,y2)
print "The regressor with best lamda for Rigid Regression is: ", clf.coef_
print 'The length of vector a in Rigid Regression is ', np.linalg.norm(clf.coef_)


# In[130]:


from sklearn import linear_model
r3 = []
r4 = []
lamda2 = []
for i in range(1,401):
    clf = linear_model.Lasso(alpha=i)
    clf.fit(x2,y2)
    r3.append(R_squared(x2,y2,clf.coef_, clf.intercept_))
    r4.append(R_squared(x1,y1,clf.coef_, clf.intercept_))
    lamda2.append(x2.shape[0]*(2)*i)
    #print(clf.coef_)
    #print(clf.intercept_)  

plt.figure()
plt.plot(lamda2,r3)
plt.plot(lamda2,r4)
plt.xlabel('the value of lamda')
plt.ylabel('R squared')
plt.title('Plot 2 : Function of lamda and R squared')
plt.savefig('figure2.pdf')


# In[131]:


print np.argmax(r4)
clf = linear_model.Lasso(alpha=(np.argmax(r4)+1))
clf.fit(x2,y2)
print "The regressor with best lamda for Lasso Regression is: ", clf.coef_
print "The best lamda for Lasso Regression is: ", lamda2[np.argmax(r4)]


# In[132]:


sqterms = np.square(x)
xall = np.insert(x, [1], sqterms, axis=1)
sqx1 = np.square(x1)
testall = np.insert(x1, [1], sqx1, axis=1)


# In[133]:


from sklearn.linear_model import LinearRegression  
regressor3 = LinearRegression()  
regressor3.fit(xall, y) 


# In[134]:


a3 = regressor3.coef_
print "The regressors of the 16 features train data is: ", regressor3.coef_
b3 = regressor3.intercept_
print "The intercept of the 16 features train data is: ", regressor3.intercept_ 


# In[135]:


print "The R^2 of the 16 features train data is: ", R_squared(xall,y,a3,b3)
print "The R^2 of the 16 features test data is: ", R_squared(testall,y1,a3,b3)

