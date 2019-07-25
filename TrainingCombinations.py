# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:12:31 2019

@author: Eduardo
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import Eval
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate


train = [105,106,107]

labels_train = []
texts_train = []
nominates_train = []

print("Generating training set")
for train_n in train:
    texts_train_path2 = "dir/speeches_"+str(train_n)+"_dwnominate_nonames.txt"
    labels_train2, texts_train2, nominates_train2 = Eval.readDataSet(texts_train_path2, 0)
    labels_train = labels_train + labels_train2
    texts_train = texts_train + texts_train2
    nominates_train = nominates_train + nominates_train2


print("Generating test set")
test_n = 112

texts_test_path = "dir/speeches_"+str(test_n)+"_dwnominate_nonames.txt"
labels_test, texts_test, nominates_test = Eval.readDataSet(texts_test_path, 0)

vectorizer = CountVectorizer(token_pattern = '[a-zA-Z]+', stop_words='english')
train_matrix = vectorizer.fit_transform(texts_train)

test_matrix = vectorizer.transform(texts_test)

print("Naive Bayes")
clf =  MultinomialNB()
clf.fit(train_matrix, labels_train)
pred = clf.predict(test_matrix)
print("Accuracy on 112th congress: " + str(Eval.Accuracy(pred, labels_test)))
"""
cv_results = cross_validate(MultinomialNB(), train_matrix, labels_train, cv = 10, verbose = 2)
test_score = cv_results['test_score']
avg = np.average(test_score)
print("10-fold cross validation accuracy: " + str(avg))
"""
print("Logistic Regression")
clf =  LogisticRegression()
clf.fit(train_matrix, labels_train)
pred = clf.predict(test_matrix)
print("Accuracy on 112th congress: " + str(Eval.Accuracy(pred, labels_test)))
"""
cv_results = cross_validate(LogisticRegression(), train_matrix, labels_train, cv = 10, verbose = 2)
test_score = cv_results['test_score']
avg = np.average(test_score)
print("10-fold cross validation accuracy: " + str(avg))
"""


