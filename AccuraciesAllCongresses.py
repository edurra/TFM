# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:35:33 2019

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


train_n = 107

texts_train_path = "directory/speeches_"+str(train_n)+"_dwnominate_nonames.txt"
labels_train, texts_train, nominates_train = Eval.readDataSet(texts_train_path, 0)

vectorizer = CountVectorizer(token_pattern = '[a-zA-Z]+', stop_words='english')
train_matrix_bow = vectorizer.fit_transform(texts_train)

vectorizer1 = TfidfVectorizer()
train_matrix_tfidf = vectorizer1.fit_transform(texts_train)

clf_nb_bow = MultinomialNB()
clf_nb_bow.fit(train_matrix_bow, labels_train)

clf_log_bow = LogisticRegression(solver='lbfgs', max_iter = 2000)
clf_log_bow.fit(train_matrix_bow, labels_train)

clf_nb_tfidf = MultinomialNB()
clf_nb_tfidf.fit(train_matrix_tfidf, labels_train)

clf_log_tfidf = LogisticRegression(solver='lbfgs', max_iter = 2000)
clf_log_tfidf.fit(train_matrix_tfidf, labels_train)

nb_bow = []
log_bow = []
nb_tfidf = []
log_tfidf = []

testn = [i for i in range(103,107)] + [i for i in range(108, 115)]

for test_n in testn:
    print(test_n)
    texts_test_path = "directory/speeches_"+str(test_n)+"_dwnominate_nonames.txt"
    labels_test, texts_test, nominates_test = Eval.readDataSet(texts_test_path, 0)
    
    test_matrix_bow = vectorizer.transform(texts_test)
    test_matrix_tfidf = vectorizer1.transform(texts_test)
    
    pred = clf_nb_bow.predict(test_matrix_bow)
    nb_bow.append(Eval.Accuracy(labels_test, pred))
    
    pred = clf_log_bow.predict(test_matrix_bow)
    log_bow.append(Eval.Accuracy(labels_test, pred))
    
    pred = clf_nb_tfidf.predict(test_matrix_tfidf)
    nb_tfidf.append(Eval.Accuracy(labels_test, pred))
    
    pred = clf_log_tfidf.predict(test_matrix_tfidf)
    log_tfidf.append(Eval.Accuracy(labels_test, pred))
    
    