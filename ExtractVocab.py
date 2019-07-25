# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:27:09 2019

@author: Eduardo
"""
from nltk.corpus import words

path = "dir/speeches_110_dwnominate_nonames.txt"
vocab_path = 'dir/vocab_110.txt'

dictionary = words.words()
f = open(path, 'r')

lines = f.readlines()

vocab = []

i = 0
for line in lines:
    i += 1
    if(i%2000 == 0):
        print(i*100/len(lines))
    text = line.split('|')[2]
    new_line = []
    text_split = text.split(" ")
    for w in text_split:
        if w.lower() not in vocab:
            vocab.append(w.lower())

actualWords = []

for w in vocab:
    if w.lower() in dictionary:
        actualWords.append(w.lower())
    else:
        for i in range(1, len(w)-1):
            if w.lower()[0:i] in dictionary:
                actualWords.append(w.lower()[0:i])
                print(w.lower()[0:i]+'---')
                if w.lower()[i:len(w)] in dictionary:
                    actualWords.append(w.lower()[i:len(w)])
                    print(w.lower()[i:len(w)])
                i = len(w)-2
            
                
f_v = open(vocab_path, 'w')
f_v.write(",".join(actualWords))
f_v.close()
