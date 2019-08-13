
# coding: utf-8

# In[ ]:


#!/usr/bin/env python


# In[1]:


from __future__ import print_function
import gzip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import svm


# In[2]:


data1 = open('/classes/ece2720/pe5/train-images-idx3-ubyte', 'rb') 
data3 = open('/classes/ece2720/pe5/t10k-images-idx3-ubyte', 'rb') 
data2 = open('/classes/ece2720/pe5/train-labels-idx1-ubyte', 'rb') 
data4 = open('/classes/ece2720/pe5/t10k-labels-idx1-ubyte', 'rb') 


# In[3]:


train1 = []
for i in range(60000*28*28+16):
    t = data1.read(1)
    t = ord(t)
    train1.append(t)
for i in range(16):
    del train1[0]


# In[4]:


test1 = []
for i in range(10000*28*28+16):
    t = data3.read(1)
    t = ord(t)
    test1.append(t)
for i in range(16):
    del test1[0]


# In[5]:


label1 = []
for i in range(60000+8):
    t = data2.read(1)
    t = ord(t)
    label1.append(t)
for i in range(8):
    del label1[0]
    
label2 = []
for i in range(10000+8):
    t = data4.read(1)
    t = ord(t)
    label2.append(t)
for i in range(8):
    del label2[0]


# In[6]:


train1 = np.reshape(train1, (60000, 784))
test1 = np.reshape(test1, (10000, 784))


# In[7]:


train_small = []
label_small = []
test_small = []
label2_small = []
for i in range(20000):
    train_small.append(train1[i])
    label_small.append(label1[i])
for i in range(10000):
    test_small.append(test1[i])
    label2_small.append(label2[i])
newtrain = []
for x in train_small:
    newtrain.append(x/255.0)
newtest = []
for x in test_small:
    newtest.append(x/255.0)


# In[8]:


from sklearn.svm import SVC
clf = SVC(gamma= 0.02, C=10)
clf.fit(newtrain, label_small)
print('SVC score RBF kernel, C=10, gamma = 0.02; Prediction Accuracy: %s' % clf.score(newtest, label2_small))


# In[9]:


import pickle
pickle.dump(clf, open('model.dat', 'wb'))

