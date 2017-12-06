#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 23:02:47 2017

@author: gowthamkommineni
"""

import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier

reviews = pd.read_csv('amazon_baby_train.csv')
reviews.shape
reviews = reviews.dropna()
reviews.shape

scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: '1' if x > 3 else '0')

reviews.groupby('rating')['review'].count().plot(kind='bar', color= ['r','g'],title='Label Distribution',  figsize = (10,6))

def splitPosNeg(Summaries):
    neg = reviews.loc[Summaries['rating']== '0']
    pos = reviews.loc[Summaries['rating']== '1']
    return [pos,neg]

[pos,neg] = splitPosNeg(reviews)


#Preprocessing
lemmatizer = nltk.WordNetLemmatizer()
stop = stopwords.words('english')
translation = str.maketrans(string.punctuation,' '*len(string.punctuation))


def preprocessing(line):
    tokens=[]
    line = line.translate(translation)
    line = nltk.word_tokenize(line.lower())
    stops = stopwords.words('english')
    stops.remove('not')
    stops.remove('no')
    line = [word for word in line if word not in stops]
    for t in line:
        stemmed = lemmatizer.lemmatize(t)
        tokens.append(stemmed)
    return ' '.join(tokens)


pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done")


data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values,neg['rating'].values))



t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)


word_features = nltk.FreqDist(t)
print(len(word_features))


#Selecting 5000 most frequent words
topwords = [fpair[0] for fpair in list(word_features.most_common(5000))]
print(word_features.most_common(25))

word_his = pd.DataFrame(word_features.most_common(200), columns = ['words','count'])

vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])


tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


ctr_features = vec.transform(data)
tr_features = tf_vec.transform(ctr_features)

tr_features.shape

clf =  MLPClassifier()
clf = clf.fit(tr_features, labels)
tfPredication = clf.predict(tr_features)
tfAccuracy = metrics.accuracy_score(tfPredication,labels)
print(tfAccuracy * 100)


## Testing Dataset.
reviews = pd.read_csv('amazon_baby_test.csv')
reviews.shape
reviews = reviews.dropna()
reviews.shape
#print(reviews.head(25))

scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: '1' if x > 3 else '0')
#print(reviews.head(25))


scores.mean()

reviews.groupby('rating')['review'].count().plot(kind='bar', color= ['r','g'],title='Label Distribution',  figsize = (10,6))

[pos,neg] = splitPosNeg(reviews)

pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done")


data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values,neg['rating'].values))


t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)
        
word_features = nltk.FreqDist(t)
print(len(word_features))

topwords = [fpair[0] for fpair in list(word_features.most_common(5002))]
print(word_features.most_common(25))

word_his = pd.DataFrame(word_features.most_common(200), columns = ['words','count'])

vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])

tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)

cte_features = vec.transform(data)
te_features = tf_vec.transform(cte_features)

te_features.shape

tePredication = clf.predict(te_features)
teAccuracy = metrics.accuracy_score(tePredication,labels)
print(teAccuracy*100)

print(metrics.classification_report(labels, tePredication))





