#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
import json
from collections import defaultdict
from sklearn import linear_model
import numpy
import math


# In[2]:


# my_root = 'C:/Users/esloe/OneDrive/Desktop/CSE 158/cse-158-HWs/datasets/'
# data = []
# f = open(my_root + 'beer_50000.json')
# for line in f:
#   data.append(eval(line))


# In[3]:


# dataTrain = data[:25000]
# dataValid = data[25000:37500]
# dataTest = data[37500:]


# In[4]:


# categoryCounts = defaultdict(int)
# for d in data:
#     categoryCounts[d['beer/style']] += 1
# categories = [c for c in categoryCounts if categoryCounts[c] > 1000]
# catID = dict(zip(list(categories),range(len(categories))))


# In[5]:


def feat(d, catID, maxLength, includeCat = True, includeReview = True, includeLength = True):
    feat = []
    if includeCat:
        # My implementation is modular such that this one function concatenates all three features together,
        # depending on which are selected
        feat.extend([0] * 12)
        if d['beer/style'] in (catID) and catID[d['beer/style']]: 
            feat[catID[d['beer/style']] - 1] = 1
    if includeReview:
        #
        ratings = [d['review/aroma'], d['review/overall'], d['review/appearance'], d['review/palate'], d['review/taste']]
        feat.extend(ratings)
    if includeLength:
        #
        len_ratio = len(d['review/text']) / maxLength
        feat.append(len_ratio)
    return feat + [1]


# In[6]:


def pipeline(reg, catID, dataTrain, dataValid, dataTest, includeCat=True, includeReview=True, includeLength=True):
    mod = linear_model.LogisticRegression(C=reg, class_weight='balanced')

    maxLength = max([len(d['review/text']) for d in dataTrain])

    Xtrain = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataTrain]
    Xvalid = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataValid]
    Xtest = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataTest]

    yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
    yValid = [d['beer/ABV'] > 7 for d in dataValid]
    yTest = [d['beer/ABV'] > 7 for d in dataTest]

    mod.fit(Xtrain,yTrain)

    pred = mod.predict(Xvalid)
    correct = pred == yValid
    TP_ = numpy.logical_and(pred, yValid)
    FP_ = numpy.logical_and(pred, numpy.logical_not(yValid))
    TN_ = numpy.logical_and(numpy.logical_not(pred), numpy.logical_not(yValid))
    FN_ = numpy.logical_and(numpy.logical_not(pred), yValid)

    TP = sum(TP_)
    FP = sum(FP_)
    TN = sum(TN_)
    FN = sum(FN_)

    vBER = 1 - 0.5*(TP / (TP + FN) + TN / (TN + FP))

    pred = mod.predict(Xtest)
    correct = pred == yTest
    TP_ = numpy.logical_and(pred, yTest)
    FP_ = numpy.logical_and(pred, numpy.logical_not(yTest))
    TN_ = numpy.logical_and(numpy.logical_not(pred), numpy.logical_not(yTest))
    FN_ = numpy.logical_and(numpy.logical_not(pred), yTest)

    TP = sum(TP_)
    FP = sum(FP_)
    TN = sum(TN_)
    FN = sum(FN_)

    tBER = 1 - 0.5*(TP / (TP + FN) + TN / (TN + FP))

    # (1) Fit the model on the training set
    # (2) Compute validation BER
    # (3) Compute test BER

    return mod, vBER, tBER


# In[7]:


### Question 1


# In[8]:


def Q1(catID, dataTrain, dataValid, dataTest):
    # No need to modify this if you've implemented the functions above
    mod, validBER, testBER = pipeline(10, catID, dataTrain, dataValid, dataTest, True, False, False)
    return mod, validBER, testBER


# In[9]:


### Question 2


# In[10]:


def Q2(catID, dataTrain, dataValid, dataTest):
    mod, validBER, testBER = pipeline(10, catID, dataTrain, dataValid, dataTest, True, True, True)
    return mod, validBER, testBER


# In[11]:


### Question 3


# In[12]:


def Q3(catID, dataTrain, dataValid, dataTest):
    # Your solution here...
    max_BER = 1
    for c in [0.001, 0.01, 0.1, 1, 10]:
        my_mod, my_validBER, my_testBER = pipeline(c, catID, dataTrain, dataValid, dataTest, True, True, True)
        if my_validBER < max_BER:
            max_BER = my_validBER
            mod, validBER, testBER = my_mod, my_validBER, my_testBER
    # Return the validBER and testBER for the model that works best on the validation set
    return mod, validBER, testBER


# In[13]:


### Question 4


# In[14]:


def Q4(C, catID, dataTrain, dataValid, dataTest):
    mod, validBER, testBER_noCat = pipeline(C, catID, dataTrain, dataValid, dataTest, False, True, True)
    mod, validBER, testBER_noReview = pipeline(C, catID, dataTrain, dataValid, dataTest, True, False, True)
    mod, validBER, testBER_noLength = pipeline(C, catID, dataTrain, dataValid, dataTest, True, True, False)
    return testBER_noCat, testBER_noReview, testBER_noLength


# In[15]:


### Question 5


# In[32]:


# my_root = 'C:/Users/esloe/OneDrive/Desktop/CSE 158/cse-158-HWs/datasets/'
# path = my_root + "/amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz"
# f = gzip.open(path, 'rt', encoding="utf8")

# header = f.readline()
# header = header.strip().split('\t')
# review_dataset = []

# pairsSeen = set()

# for line in f:
#     fields = line.strip().split('\t')
#     d = dict(zip(header, fields))
#     ui = (d['customer_id'], d['product_id'])
#     if ui in pairsSeen:
#         continue
#     pairsSeen.add(ui)
#     d['star_rating'] = int(d['star_rating'])
#     d['helpful_votes'] = int(d['helpful_votes'])
#     d['total_votes'] = int(d['total_votes'])
#     review_dataset.append(d)
# reviewDataTrain = review_dataset[:int(len(review_dataset)*0.9)]
# reviewDataTest = review_dataset[int(len(review_dataset)*0.9):]
# usersPerItem = defaultdict(set) # Maps an item to the users who rated it
# itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
# itemNames = {}
# ratingDict = {} # To retrieve a rating for a specific user/item pair
# reviewsPerUser = defaultdict(list)

# for d in reviewDataTrain:
#     user,item = d['customer_id'], d['product_id']
#     usersPerItem[item].add(user)
#     itemsPerUser[user].add(item)
#     reviewsPerUser[user].append(d)

# for d in review_dataset:
#     user,item = d['customer_id'], d['product_id']
#     ratingDict[(user,item)] = d['star_rating']
#     itemNames[item] = d['product_title']


# In[17]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom


# In[18]:


def mostSimilar(i, N, usersPerItem):
    # Implement...
    # Should be a list of (similarity, itemID) pairs
    similarities = []
    base_users = usersPerItem[i]
    for i2 in usersPerItem: #Keys are itemID, dictionary values are users who bought.
        if i2 == i:
            continue
        similarity = Jaccard(base_users,usersPerItem[i2])

        similarities.append((similarity,i2))
    similarities.sort(reverse=True)
    return similarities[:N]


# In[19]:


### Question 6


# In[20]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[21]:


def getMeanRating(dataTrain):
    # Implement...
    mean = sum(d['star_rating'] for d in dataTrain) / len(dataTrain)
    return mean

def getUserAverages(itemsPerUser, ratingDict):
    # Implement (should return a dictionary mapping users to their averages)
    userAverages = {}
    for user in itemsPerUser:
        rating_sum = 0
        for i in itemsPerUser[user]:
            rating_sum += ratingDict[(user,i)]
        userAverages[user] = rating_sum / len(itemsPerUser[user])
    return userAverages

def getItemAverages(usersPerItem, ratingDict):
    # Implement...
    itemAverages = {}
    for item in usersPerItem:
        rating_sum = 0
        for u in usersPerItem[item]:
            rating_sum += ratingDict[(u,item)]
        itemAverages[item] = rating_sum / len(usersPerItem[item])
    return itemAverages


# In[79]:


def predictRating(user,item,ratingMean,reviewsPerUser,usersPerItem,itemsPerUser,userAverages,itemAverages):
    # Solution for Q6, should return a rating
    if item not in itemAverages:
        return ratingMean

    ratings = []
    sims = []
    for d in reviewsPerUser[user]:
        j = d['product_id']
        if j == item: continue
        ratings.append(d['star_rating'] - itemAverages[j])
        sims.append(Jaccard(usersPerItem[item],usersPerItem[j]))

    if sum(sims) == 0:
        return itemAverages[item]

    if (sum(sims) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,sims)]
        return itemAverages[item] + sum(weightedRatings) / sum(sims)
    else:
        # User hasn't rated any similar items
        return ratingMean


# In[23]:


### Question 7


# In[80]:


def predictRatingQ7(user,item,ratingMean,reviewsPerUser,usersPerItem,itemsPerUser,userAverages,itemAverages):
    # Solution for Q7, should return a rating
    if item not in itemAverages:
        return ratingMean

    if len(usersPerItem.get(item, [])) < 4:
        return ratingMean

    ratings = []
    sims = []
    for d in reviewsPerUser[user]:
        j = d['product_id']
        if j == item: continue
        ratings.append(d['star_rating'] - itemAverages[j])
        sims.append(Jaccard(usersPerItem[item],usersPerItem[j]))

    if sum(sims) == 0:
        return itemAverages.get(item, ratingMean)

    if (sum(sims) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,sims)]
        return itemAverages[item] + sum(weightedRatings) / sum(sims)
    else:
        # User hasn't rated any similar items
        return ratingMean


# In[65]:


# ypred = [predictRatingQ7(d['customer_id'],d['product_id'],ratingMean,reviewsPerUser,usersPerItem,itemsPerUser,userAverages,itemAverages) for d in reviewDataTest]


# In[34]:


# ratingMean = getMeanRating(reviewDataTrain)
# # userAverages = getUserAverages(itemsPerUser, ratingDict)
# itemAverages = getItemAverages(usersPerItem, ratingDict)


# In[66]:


# y = [d['star_rating']for d in reviewDataTest]


# In[67]:


# MSE1 = MSE(y,ypred)


# In[68]:


# print(MSE1)


# In[40]:


# ypred2 = [predictRating(d['customer_id'],d['product_id'],ratingMean,reviewsPerUser,usersPerItem,itemsPerUser,userAverages,itemAverages) for d in reviewDataTest]


# In[41]:


# MSE2 = MSE(y,ypred2)
# print(MSE2)


# In[42]:


# y2 = [ratingMean for d in reviewDataTest]


# In[44]:


# MSE(y2,y)


# In[ ]:




