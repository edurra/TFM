# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 18:36:18 2019

@author: Eduardo
"""

import CreateMatrix
import matplotlib.pyplot as plt
import math

train_n = 110
test_n = 112
vocab_path = 'directory/vocab.txt'

new_train = "directory/speeches_"+str(train_n)+"_dwnominate_features_nonames.txt"
new_test = "directory/speeches_"+str(test_n)+"_dwnominate_features_nonames.txt"

texts_train_path = "directory/speeches_"+str(train_n)+"_dwnominate_nonames.txt"
texts_test_path = "directory/speeches_"+str(test_n)+"_dwnominate_nonames.txt"

train_pos = "directory/speeches_"+str(train_n)+"_dwnominate_POS_nonames.txt"
test_pos = "directory/speeches_"+str(test_n)+"_dwnominate_POS_nonames.txt"

train_pos_gram = "directory/speeches_"+str(train_n)+"_dwnominate_POS_2gram_nonames.txt"
test_pos_gram = "directory/speeches_"+str(test_n)+"_dwnominate_POS_2gram_nonames.txt"

new_train_ngrams = "directory/speeches_"+str(train_n)+"_dwnominate_n_3_nonames.txt"
new_test_ngrams = "directory/speeches_"+str(test_n)+"_dwnominate_n_3_nonames.txt"

train_lda = 'directory/speeches_'+str(train_n)+'_dwnominate_lda_14_nonames.txt'
test_lda = 'directory/speeches_'+str(test_n)+'_dwnominate_lda_14_nonames.txt'

tpos = []
tneg = []
tmax = 0.913
#tmin = -0.643
tmin = -0.682
for i in range(0,10):
    tpos.append((i)*tmax/10)
    
for i in range(0,10):
    tneg.append((i)*tmin/10)

tpostest = []
tnegtest = []
tmaxtest = 0.913
tmintest = -0.643
for i in range(0,10):
    tpostest.append((i)*tmaxtest/10)
    
for i in range(0,10):
    tnegtest.append((i)*tmintest/10)
    
nb_list = []
log_list = []
labels_test_list = []
labels_train_list = []
for i in range(0, len(tpos)):
    print("-------------------------------------------")
    print("Thresholds: " + str(tneg[i]) + "|" + str(tpos[i]))
    nb_ac, log_ac, labels_test, labels_train =  CreateMatrix.TestModel(new_train, new_test, texts_train_path, texts_test_path, train_pos, test_pos, new_train_ngrams, new_test_ngrams, train_lda, test_lda, thresholdPos = tpos[i], thresholdNeg = tneg[i], thresholdPosTest = tpostest[i], thresholdNegTest =tnegtest[i], subsample=False, removeCenter=True, BoW = False, charNgrams = False, POS = False, features = False, POSgrams = False, tfidf = True, binary = False, lda = False, addToTrain = [109,111])
    nb_list.append(nb_ac)
    log_list.append(log_ac)
    labels_test_list.append(labels_test)
    labels_train_list.append(labels_train)

baseline = []
labels_test_pos = []
labels_test_neg = []
for t in labels_test_list:
    labels_test_pos.append(len([x for x in t if x == 1.0]))
    labels_test_neg.append(len([x for x in t if x == -1.0]))
    
labels_test_pos[0]
for i in range(0, len(labels_test_list)):
    b1 = labels_test_pos[i]/len(labels_test_list[i])
    b2 = labels_test_neg[i]/len(labels_test_list[i])
    baseline.append(max(b1,b2))
    

#xLabels = [str(round(tneg[i],2))+'\n'+str(round(tpos[i],2)) for i in range(len(tpos))]
xLabels = [math.floor((tpos[i]-tneg[i])/(tmax-tmin)*10)/10 for i in range(0,10)]
plt.plot(xLabels[0:len(xLabels)], log_list, color = 'green', label = 'Logistic Regression')
plt.plot(xLabels[0:len(xLabels)], nb_list, color = 'red', label = 'Naive Bayes')
plt.legend(loc='best')
plt.xlabel('Percentage of DW-Nominate removed')
plt.ylabel('Accuracy')
plt.xticks(xLabels)