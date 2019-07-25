# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:51:07 2019

@author: Eduardo
"""
import Eval
import operator
import datetime

train_n = 107
test_n = 109

texts_train_path = "dir/speeches_"+str(train_n)+"_dwnominate_nonames.txt"
texts_test_path = "dir/speeches_"+str(test_n)+"_dwnominate_nonames.txt"

n = 3
new_train_ngrams = "dir/speeches_"+str(train_n)+"_dwnominate_n_"+str(n)+"_nonames.txt"
new_test_ngrams = "dir/speeches_"+str(test_n)+"_dwnominate_n_"+str(n)+"_nonames.txt"

nFeatures = 500
ngrams = []

labels_train, texts_train, nominates_train = Eval.readDataSet(texts_train_path, 0)
labels_test, texts_test, nominates_test = Eval.readDataSet(texts_test_path, 0)
i = 0

print(datetime.datetime.now())
ngrams_dict = {}
for t in texts_train:
    if(i%2000 == 0):
        print(i*100/len(texts_train))
        print("Generating ngrams")

    i+=1
    
    for j in range(0, len(t)-n+1):
        ng = t[j:j+n]
        if ng in ngrams_dict.keys():
            ngrams_dict[ng] += 1
        else:
            ngrams_dict[ng] = 1

dic_sorted = sorted(ngrams_dict.items(), key = operator.itemgetter(1))
ngrams = []
for j in range(len(dic_sorted)-nFeatures,len(dic_sorted)):
    
    ngrams.append(dic_sorted[j][0])
        
train_rows = []
test_rows = []


new_train_file = open(new_train_ngrams, 'w')
new_train_file.write(",".join(ngrams)+"\n")
i = 0
print(datetime.datetime.now())
for t in texts_train:
    if(i%2000 == 0):
        print(i*100/len(texts_train))
        print("Processing training file")
    i+=1
    text_count = [0]*len(ngrams)
    for j in range(0, len(t)-n+1):
        ng = t[j:j+n]
        if ng in ngrams:
            text_count[ngrams.index(ng)] += 1
    new_train_file.write(",".join([str(i) for i in text_count]) + "\n")
new_train_file.close()


new_test_file = open(new_test_ngrams, 'w')
new_test_file.write(",".join(ngrams)+"\n")
i = 0
print(datetime.datetime.now())
for t in texts_test: 
    if(i%2000 == 0):
        print(i*100/len(texts_test))
        print("Processing test file")
    i+=1
    text_count = [0]*len(ngrams)
    for j in range(0, len(t)-n+1):
        ng = t[j:j+n]
        if ng in ngrams:
            text_count[ngrams.index(ng)] += 1
    new_test_file.write(",".join([str(i) for i in text_count]) + "\n")
new_test_file.close()
