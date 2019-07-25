# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:51:12 2019

@author: Eduardo
"""

import Eval
import operator
import datetime
from nltk.tokenize import RegexpTokenizer

texts_train_path = "directory/speeches_110_dwnominate_nonames.txt"
texts_test_path = "directory/speeches_112_dwnominate_nonames.txt"

n = 3
new_train_ngrams = "directory/speeches_110_dwnominate_n_begend_nonames.txt"
new_test_ngrams = "directory/speeches_112_dwnominate_n_begend_nonames.txt"

nFeatures = 500
ngrams = []

labels_train, texts_train, nominates_train = Eval.readDataSet(texts_train_path, 0)
labels_test, texts_test, nominates_test = Eval.readDataSet(texts_test_path, 0)
i = 0

tokenizer = RegexpTokenizer(r'\w+')

print(datetime.datetime.now())
ngrams_dict = {}
for t in texts_train:
    if(i%2000 == 0):
        print(i*100/len(texts_train))
        print("Generating ngrams")

    i+=1
    
    words = tokenizer.tokenize(t)
    
    for w in words:
        if len(w) == 3:
            if(w.lower() in ngrams_dict.keys()):
                ngrams_dict[w.lower()] = ngrams_dict[w.lower()] + 1
            else:
                ngrams_dict[w.lower()] = 1
        if len(w) > 3:
            beg = w.lower()[0:3]
            end = w.lower()[len(w)-3:len(w)]
            
            if beg in ngrams_dict.keys():
                ngrams_dict[beg] = ngrams_dict[beg] + 1
            else:
                ngrams_dict[beg] = 1
                
            if end in ngrams_dict.keys():
                ngrams_dict[end] = ngrams_dict[end] + 1
            else:
                ngrams_dict[end] = 1

dic_sorted = sorted(ngrams_dict.items(), key = operator.itemgetter(1))
ngrams = []

nFeatures = min(nFeatures, len(ngrams_dict.keys()))

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
    
    words = tokenizer.tokenize(t)
    
    for w in words:
        if len(w) == 3:
            if(w.lower() in ngrams):
                text_count[ngrams.index(w.lower())] += 1
        if len(w) > 3:
            beg = w.lower()[0:3]
            end = w.lower()[len(w)-3:len(w)]
            
            if beg in ngrams:
                text_count[ngrams.index(beg)] += 1
                
            if end in ngrams:
                text_count[ngrams.index(end)] += 1
                
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
    words = tokenizer.tokenize(t)
    
    for w in words:
        if len(w) == 3:
            if(w.lower() in ngrams):
                text_count[ngrams.index(w.lower())] += 1
        if len(w) > 3:
            begg = w.lower()[0:3]
            end = w.lower()[len(w)-3:len(w)]
            
            if begg in ngrams:
                text_count[ngrams.index(begg)] += 1
                
            if end in ngrams:
                text_count[ngrams.index(end)] += 1
    new_test_file.write(",".join([str(i) for i in text_count]) + "\n")
new_test_file.close()