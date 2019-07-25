# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:56:19 2019

@author: Eduardo
"""
import Eval
import nltk
from nltk.tokenize import RegexpTokenizer

train_n = 107
test_n = 109

texts_train_path = "directory/speeches_"+str(train_n)+"_dwnominate_nonames_aux.txt"
texts_test_path = "directory/speeches_"+str(test_n)+"_dwnominate_nonames_aux.txt"

new_train_pos = "directory/speeches_"+str(train_n)+"_dwnominate_POS_nonames.txt"
new_test_pos = "directory/speeches_"+str(test_n)+"_dwnominate_POS_nonames.txt"


labels_train, texts_train, nominates_train = Eval.readDataSet(texts_train_path, 0)
labels_test, texts_test, nominates_test = Eval.readDataSet(texts_test_path, 0)

train_rows = []
tokenizer = RegexpTokenizer(r'\w+')
tags = []
i = 0
print("Calculating POS")
count = 0
for t in texts_train:
    print(i*100/len(texts_train))
    i+=1
    if(count < 300):
        count += 1
        words = tokenizer.tokenize(t)
        pos_tags = nltk.pos_tag(words)
        for p in pos_tags:
            if p[1] not in tags:            
                tags.append(p[1])
                count = 0

train_file = open(new_train_pos, 'w')
train_file.write(",".join(tags)+"\n")

print("Processing training file")
i = 0
for t in texts_train:
    if(i%10000 == 0):
        print(i*100/len(texts_train))
        print("Processing training file")
    i+=1
    words = tokenizer.tokenize(t)
    pos_tags = nltk.pos_tag(words)
    row_tags = {key:0 for key in tags}
    for p in pos_tags:
        if p[1] in tags:
            row_tags[p[1]] += 1
    train_file.write(",".join([str(x) for x in list(row_tags.values())])+"\n")
train_file.close()  

test_file = open(new_test_pos, 'w')
test_file.write(",".join(tags)+"\n")


i = 0
for t in texts_test:
    if(i%10000 == 0):
        print("Processing test file")
        print(i*100/len(texts_test))
    i+=1
    words = tokenizer.tokenize(t)
    pos_tags = nltk.pos_tag(words)
    row_tags = {key:0 for key in tags}
    for p in pos_tags:
        if p[1] in tags:
            row_tags[p[1]] += 1
    test_file.write(",".join([str(x) for x in list(row_tags.values())])+"\n")
test_file.close()  