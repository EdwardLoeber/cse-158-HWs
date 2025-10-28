#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
import json
import dateutil.parser
import random


# In[2]:


from collections import defaultdict
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy
import math


# In[3]:


# my_root = 'C:/Users/esloe/OneDrive/Desktop/CSE 158/cse-158-HWs/datasets/'


# In[4]:


### Question 1
# my_root = 'C:/Users/esloe/OneDrive/Desktop/CSE 158/cse-158-HWs/datasets/'
# dataset = []
# f = gzip.open(my_root + 'fantasy_10000.json.gz')
# for line in f:
#   dataset.append(json.loads(line))


# In[5]:


def getMaxLen(dataset):
    # Find the longest review (number of characters)
    maxLen = 0
    for d in dataset:
        maxLen = len(d['review_text']) if (len(d['review_text']) > maxLen) else maxLen
    return maxLen


# In[6]:


def featureQ1(datum, maxLen):
    # Feature vector for one data point
    length_ratio = len(datum['review_text']) / maxLen
    return [1, length_ratio]


# In[7]:


def Q1(dataset):
    # Implement...
    maxLen = getMaxLen(dataset)
    ratings = [d['rating'] for d in dataset]
    y = numpy.array(ratings).reshape(-1,1)
    X = numpy.array([featureQ1(d, maxLen) for d in dataset])

    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X,y)
    theta = model.coef_

    MSE = numpy.mean((y - X @ theta.T) ** 2)
    theta = theta[0]

    return theta, MSE


# In[8]:


### Question 2


# In[9]:


def featureQ2(datum, maxLen):
    # Implement (should be 1, length feature, day feature, month feature)
    feature_list = featureQ1(datum, maxLen)
    feature_list.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # could do 0 times #features right? idc
    t = dateutil.parser.parse(datum['date_added']) 
    day = t.weekday() # monday = 0
    month = t.month - 1 # for 0-indexing (normally, january = 1)

    # one-hot encoding
    if (day>0):
        feature_list[1+day] = 1 
    if (month>0):
        feature_list[1+6+month] = 1
    return feature_list


# In[10]:


def Q2(dataset):
    # Implement (note MSE should be a *number*, not e.g. an array of length 1)
    maxLen = getMaxLen(dataset)
    Y2 = numpy.array([d['rating'] for d in dataset]).reshape(-1,1)
    X2 = numpy.array([featureQ2(datum, maxLen) for datum in dataset])

    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X2,Y2)
    theta = model.coef_

    MSE2 = numpy.mean((Y2 - X2 @ theta.T) ** 2)
    theta = theta[0]

    return X2, Y2, MSE2


# In[11]:


def featureQ3(datum, maxLen):
    # Implement (should be 1, length feature, day feature, month feature)
    feature_list = featureQ1(datum, maxLen)
    t = dateutil.parser.parse(datum['date_added']) 
    day = t.weekday() # monday = 0
    month = t.month # january = 1
    feature_list.extend([day,month])

    return feature_list


# In[12]:


def Q3(dataset):
    # Implement
    maxLen = getMaxLen(dataset)
    Y3 = numpy.array([d['rating'] for d in dataset]).reshape(-1,1)
    X3 = numpy.array([featureQ3(datum, maxLen) for datum in dataset])

    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X3,Y3)
    theta = model.coef_

    MSE3 = numpy.mean(((Y3 - X3 @ theta.T) ** 2))
    theta = theta[0]

    return X3, Y3, MSE3


# In[13]:


def Q4(dataset):
    # Implement
    maxLen = getMaxLen(dataset)
    y = numpy.array([d['rating'] for d in dataset]).reshape(-1,1)
    X_onehot = numpy.array([featureQ2(datum,maxLen) for datum in dataset])
    X_dateasis = numpy.array([featureQ3(datum,maxLen) for datum in dataset])

    X_onehot_train, X_onehot_test, y_train, y_test = train_test_split(X_onehot, y, test_size=0.5,shuffle=False)
    X_dateasis_train, X_dateasis_test, y_train, y_test = train_test_split(X_onehot, y, test_size=0.5,shuffle=False)

    model_onehot = linear_model.LinearRegression(fit_intercept=False)
    model_asis = linear_model.LinearRegression(fit_intercept=False)

    model_onehot.fit(X_onehot_train,y_train)
    model_asis.fit(X_dateasis_train,y_train)

    y_onehot_pred = model_onehot.predict(X_onehot_test)
    y_asis_pred = model_asis.predict(X_dateasis_test)

    test_mse3 = numpy.mean((y_test - y_onehot_pred) ** 2)
    test_mse2 = numpy.mean((y_test - y_asis_pred) ** 2)

    return test_mse2, test_mse3


# In[14]:


### Question 5


# In[15]:


# my_root = 'C:/Users/esloe/OneDrive/Desktop/CSE 158/cse-158-HWs/datasets/'
# datasetB = []
# f = open(my_root + 'beer_50000.json')
# for line in f:
#   datasetB.append(eval(line))


# In[17]:


def featureQ5(datum):
    a = [1, len(datum['review/text'])]
    return a
    # Implement


# In[18]:


def Q5(dataset, feat_func):
    # Implement
    y = [d['review/overall'] >= 4 for d in dataset]
    X = [feat_func(d) for d in dataset]

    model = linear_model.LogisticRegression(class_weight='balanced')
    model.fit(X,y)

    predictions = model.predict(X)

    TP, TN, FP, FN, i = 0,0,0,0,0

    while(i < len(y)):
        if y[i] and predictions[i]:
            TP += 1
        elif y[i] and not predictions[i]:
            FN += 1
        elif not y[i] and not predictions[i]:
            TN += 1
        else:
            FP += 1
        i += 1
    BER = 1/2 * ((FP/(FP+TN) + FN/(FN+TP)))

    return TP, TN, FP, FN, BER


# In[19]:


### Question 6


# In[20]:


def Q6(dataset):
    # Implement
    y = [d['review/overall'] >= 4 for d in dataset]
    X = [featureQ5(d) for d in dataset]

    model = linear_model.LogisticRegression(class_weight='balanced')
    model.fit(X,y)

    confidences = model.decision_function(X)
    sortedByConfidence = list(zip(confidences,y))
    sortedByConfidence.sort(reverse=True)

    precs = []
    specified_precisions = [1,100,1000,10000]
    for k in specified_precisions:
        retrievedLabels = [x[1] for x in sortedByConfidence[:k]]
        precs.append(sum(retrievedLabels) / len(retrievedLabels))

    return precs


# In[21]:


### Question 7


# In[27]:


def featureQ7(datum):
    a = [1, datum['review/palate'], datum['review/taste']]
    return a
    # Implement (any feature vector which improves performance over Q5)


# In[ ]:




