
# coding: utf-8

# In[ ]:


#!/usr/bin/env python


# In[49]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import pickle


# In[50]:


data3 = open('/classes/ece2720/pe5/t10k-images-idx3-ubyte', 'rb') 


# In[51]:


test1 = []
for i in range(10000*28*28+16):
    t = data3.read(1)
    t = ord(t)
    test1.append(t)
for i in range(16):
    del test1[0]


# In[52]:


test1 = np.reshape(test1, (10000, 784))


# In[53]:


newtest = []
for x in test1:
    newtest.append(x/255.0)


# In[54]:


clf = pickle.load(open('model.dat', 'rb'))
guesses = clf.predict(newtest)
s = ','.join([t for t in guesses.astype(str)])
f = open('digits.csv', 'w')
f.write(s)
f.close()

print ('Export digits.csv file')
