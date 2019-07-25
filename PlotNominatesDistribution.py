# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:43:16 2019

@author: Eduardo
"""
import Eval
import matplotlib.pyplot as plt

texts_train_path = "dir/speeches_111_dwnominate_nonames.txt"
texts_test_path = "dir/speeches_112_dwnominate_nonames.txt"

labels_train, texts_train, nominates_train = Eval.readDataSet(texts_train_path, 0)
labels_test, texts_test, nominates_test = Eval.readDataSet(texts_test_path, 0)

indices = [l/10 for l in list(range(-10,10))]

train_nominates = {}

for i in indices:
    train_nominates[i] = 0

for n in nominates_train:
    for i in indices:
        if n>=i and n < i+0.1:
            train_nominates[i] += 1

plt.bar(x = indices, height = train_nominates.values(), align = 'edge', edgecolor= 'black', width = 0.1)
plt.xlabel("Nominate")
plt.ylabel("Number of samples")



