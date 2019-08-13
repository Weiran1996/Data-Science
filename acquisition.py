
#!/usr/bin/env python
import os
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import csv
import codecs
import math
import pandas as pd
import scipy
import scipy.stats
from scipy.stats import norm
from scipy.stats import probplot
from scipy.special import ndtri


# In[255]:


f1 = codecs.open('unicode1.dat', encoding= 'utf-8')
f2 = codecs.open('unicode2.dat', encoding= 'utf-32-le')
f3 = codecs.open('unicode3.dat', encoding= 'ASCII')
f4 = codecs.open('unicode4.dat', encoding= 'utf-16')
f5 = codecs.open('unicode5.dat', encoding= 'utf-32-be')


# In[256]:


s1 = f1.readline()
s2 = f2.readline()
s3 = f3.readline()
s4 = f4.readline()
s5 = f5.readline()
print s1
print s2
print s3
print s4
print s5


# In[257]:


print 'The size of unicode1 is: ', os.path.getsize('unicode1.dat')
print 'The size of unicode1 is: ', os.path.getsize('unicode2.dat')
print 'The size of unicode1 is: ', os.path.getsize('unicode3.dat')
print 'The size of unicode1 is: ', os.path.getsize('unicode4.dat')
print 'The size of unicode1 is: ', os.path.getsize('unicode5.dat')


# In[258]:


list0=[]
with open('synthetic.csv', 'rb') as file1:
    content1 = csv.reader(file1)
    for i in content1:
        list0.append(i)
list1 = list0[0]
list1 = [float(x) for x in list1]


# In[259]:


sum = 0.0
for i in range(3000):
    sum = sum + list1[i]
mu = sum/3000


# In[260]:


total = 0.0
for i in list1:
    total = total + (mu - i)**2
total = total/3000
sigma = math.sqrt(total)
print mu
print sigma


# In[261]:


plt.hist(list1, bins=50)
plt.ylabel('Frequency');
plt.xlabel('Data')
plt.title('Historgram of Data')
plt.savefig('figure1.pdf')


# In[262]:


plt.figure()
scipy.stats.probplot(list1,dist = 'norm', plot=plt)
plt.savefig('figure2.pdf')


# In[263]:


for i in list1:
    if (abs(i-mu)/sigma) > Dmax:
        print i
print mu


# In[264]:


mean1 = np.mean(list1)
std1 = np.std(list1)
for i in range(3000):
        fi = scipy.stats.norm(mean1,std1).cdf(list1[i])
        if 2*3000*fi<1:
            print list1[i]


# In[265]:


list2=[]
with open('passengers.csv', 'rb') as file2:
    content2 = csv.reader(file2)
    for i in content2:
        list2.append(i)


# In[266]:


list2 = list2[1:]
list_age = []
for i in range(891):
    list_age.append(list2[i][5])


# In[267]:


temp=[]
for i in list_age:
    if i!='':
        temp.append(i)
list_age = temp
list_age = [float(x) for x in list_age]

plt.hist(list_age, bins=20)
plt.ylabel('Frequency');
plt.xlabel('Range of Age')
plt.title('Historgram of Age')
plt.savefig('figure3.pdf')


# In[268]:


list_fare=[]
for i in range(891):
    list_fare.append(list2[i][9])
temp=[]
for i in list_fare:
    if i!='':
        temp.append(i)
list_fare = temp
list_fare = [float(x) for x in list_fare]

plt.hist(list_fare, bins=30)
plt.ylabel('Frequency');
plt.xlabel('Range of Fare')
plt.title('Historgram of Fare')
plt.savefig('figure4.pdf')


# In[269]:


print np.mean(list_age)
print np.std(list_age)
print np.mean(list_fare)
print np.std(list_fare)


# In[270]:


plt.figure()
scipy.stats.probplot(list_age,dist = 'norm', plot=plt)
plt.savefig('figure5.pdf')


# In[271]:


plt.figure()
scipy.stats.probplot(list_fare,dist = 'norm', plot=plt)
plt.savefig('figure6.pdf')


# In[272]:


float(np.sum(table.loc[table['Sex']=='female']['Survived']))/len(table.loc[table['Sex']=='female']['Survived'])


# In[273]:


table=pd.read_csv('passengers.csv')

table


# In[274]:


float(np.sum(table.loc[table['Sex']=='female']['Survived']))/len(table.loc[table['Sex']=='female']['Survived'])


# In[275]:


float(np.sum(table.loc[table['Sex']=='male']['Survived']))/len(table.loc[table['Sex']=='male']['Survived'])


# In[276]:


float(np.sum(table.loc[table['Pclass']==1]['Survived']))/len(table.loc[table['Pclass']==1]['Survived'])


# In[277]:


float(np.sum(table.loc[table['Pclass']==3]['Survived']))/len(table.loc[table['Pclass']==3]['Survived'])


# In[278]:


float(np.sum(table.loc[(table['Pclass']==1) & (table['Sex']=='male')]['Survived']))/len(table.loc[(table['Pclass']==1) & (table['Sex']=='male')])


# In[279]:


float(np.sum(table.loc[(table['Pclass']==3) & (table['Sex']=='female')]['Survived']))/len(table.loc[(table['Pclass']==3) & (table['Sex']=='female')])


# In[280]:


float(np.sum(table.loc[table['Fare']>100]['Survived']))/len(table.loc[table['Fare']>100]['Survived'])


# In[281]:


float(np.sum(table.loc[table['Fare']<50]['Survived']))/len(table.loc[table['Fare']<50]['Survived'])


# In[282]:


float(np.sum(table.loc[table['Parch']!=0]['Survived']))/len(table.loc[table['Parch']!=0]['Survived'])

