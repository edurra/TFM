# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:00:11 2019

@author: Eduardo
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.preprocessing import normalize
import numpy as np
import Eval
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import nltk
import FeaturesReader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sklearn.metrics import confusion_matrix




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

def TestModel(new_train, new_test, texts_train_path, texts_test_path, train_pos, test_pos, new_train_ngrams, new_test_ngrams, train_lda, test_lda, thresholdPos = 0.2, thresholdNeg = -0.2, thresholdPosTest = 0.2, thresholdNegTest = -0.2, subsample=False, removeCenter=True, BoW = True, charNgrams = False, POS = False, features = False, POSgrams = False, tfidf = False, binary = False, lda = False, addToTrain = None):
    
    names = []
    
    labels_train, texts_train, nominates_train = Eval.readDataSet(texts_train_path, 0)
    labels_test, texts_test, nominates_test = Eval.readDataSet(texts_test_path, 0)
    
    if addToTrain:
        for i in addToTrain:
            texts_train_path2 = "C:/Users/Eduardo/Desktop/2 cuatri IIT/TFM/Datasets/hein-daily/hein-daily/longTexts/speeches_"+str(i)+"_dwnominate_nonames.txt"
            labels_train2, texts_train2, nominates_train2 = Eval.readDataSet(texts_train_path2, 0)
            labels_train = labels_train + labels_train2
            texts_train = texts_train + texts_train2
            nominates_train = nominates_train + nominates_train2
    
    """
    train_pos_3gram_file = open(train_pos_3gram, 'r')
    train_pos_3gram_file_list = train_pos_3gram_file.readlines() 
    pos_3gram_names = train_pos_3gram_file_list[0].split(",")
    train_pos_3gram_file_list.pop(0)
    test_pos_3gram_file = open(test_pos_3gram,'r')
    test_pos_3gram_file_list = test_pos_3gram_file.readlines()
    test_pos_3gram_file_list.pop(0)
    """    
      
    if features:
        print("Reading feature files")
        train, test, train_labels, test_labels, feature_names = FeaturesReader.readFeatures(new_train, new_test) 

        train_matrix = np.matrix(train)
        test_matrix = np.matrix(test)
        
        names = names + feature_names
    
    if POS:
        print("Reading POS files")
        
        train_pos_file = open(train_pos, 'r')
        train_pos_file_list = train_pos_file.readlines() 
        pos_names = train_pos_file_list[0].split(",")
        train_pos_file_list.pop(0)
        test_pos_file = open(test_pos,'r')
        test_pos_file_list = test_pos_file.readlines()
        test_pos_file_list.pop(0)
        
        pos_train_rows = []
        for line in train_pos_file_list:
            line = line.replace("\n",'')
            pos_train_rows.append([int(r) for r in line.split(',')])
        train_pos_file.close()
        
        
        pos_test_rows = []
        for line in test_pos_file_list:
            line = line.replace("\n",'')
            pos_test_rows.append([int(r) for r in line.split(',')])
        test_pos_file.close()
            
        if features:
            train_matrix = np.concatenate([train_matrix, np.matrix(pos_train_rows)], axis = 1)
            test_matrix = np.concatenate([test_matrix, np.matrix(pos_test_rows)], axis = 1)
        else:
            train_matrix = np.matrix(pos_train_rows)
            test_matrix = np.matrix(pos_test_rows)
        names = names + pos_names   
        
    
    if charNgrams:
        print("Reading ngram files")
        train_ngram_file = open(new_train_ngrams, 'r')
        train_ngram_file_list = train_ngram_file.readlines()
        ngram_names = train_ngram_file_list[0].split(",")
        train_ngram_file_list.pop(0)
        test_ngram_file = open(new_test_ngrams, 'r')
        test_ngram_file_list = test_ngram_file.readlines()
        test_ngram_file_list.pop(0)
        
        lines = train_ngram_file_list
        ngram_train_rows = []
        for line in lines:
            line = line.replace("\n",'')
            ngram_train_rows.append([int(r) for r in line.split(',')])
        train_ngram_file.close()
        
        
        lines = test_ngram_file_list
        ngram_test_rows = []
        for line in lines:
            line = line.replace("\n",'')
            
            ngram_test_rows.append([int(r) for r in line.split(',')])
        test_ngram_file.close()
        
        if (features or POS):
            train_matrix = np.concatenate([train_matrix, np.matrix(ngram_train_rows)], axis = 1)
            test_matrix = np.concatenate([test_matrix, np.matrix(ngram_test_rows)], axis = 1)
        else:
            train_matrix = np.matrix(ngram_train_rows)
            test_matrix = np.matrix(ngram_test_rows)
        names = names + ngram_names
            
    if POSgrams:
        print("Reading POS n gram files")
        
        train_pos_gram_file = open(train_pos_gram, 'r')
        train_pos_gram_file_list = train_pos_gram_file.readlines() 
        pos_gram_names = train_pos_gram_file_list[0].split(",")
        train_pos_gram_file_list.pop(0)
        test_pos_gram_file = open(test_pos_gram,'r')
        test_pos_gram_file_list = test_pos_gram_file.readlines()
        test_pos_gram_file_list.pop(0)
        
        pos_gram_train_rows = []
        for line in train_pos_gram_file_list:
            line = line.replace("\n",'')
            pos_gram_train_rows.append([int(r) for r in line.split(',')])
        train_pos_gram_file.close()
        
        
        pos_gram_test_rows = []
        for line in test_pos_gram_file_list:
            line = line.replace("\n",'')
            pos_gram_test_rows.append([int(r) for r in line.split(',')])
        test_pos_gram_file.close()
            
        if (features or POS or charNgrams):
            train_matrix = np.concatenate([train_matrix, np.matrix(pos_gram_train_rows)], axis = 1)
            test_matrix = np.concatenate([test_matrix, np.matrix(pos_gram_test_rows)], axis = 1)
        else:
            train_matrix = np.matrix(pos_gram_train_rows)
            test_matrix = np.matrix(pos_gram_test_rows)
        names = names + pos_gram_names
        
        """
        pos_3gram_train_rows = []
        for line in train_pos_3gram_file_list:
            line = line.replace("\n",'')
            pos_3gram_train_rows.append([int(r) for r in line.split(',')])
        train_pos_3gram_file.close()
        
        
        pos_3gram_test_rows = []
        for line in test_pos_3gram_file_list:
            line = line.replace("\n",'')
            pos_3gram_test_rows.append([int(r) for r in line.split(',')])
        test_pos_3gram_file.close()
            

        train_matrix = np.concatenate([train_matrix, np.matrix(pos_gram_train_rows)], axis = 1)
        test_matrix = np.concatenate([test_matrix, np.matrix(pos_gram_test_rows)], axis = 1)
        
        names = names + pos_3gram_names
        """
        
    if lda:
        print("Reading lda files")
        
        train_lda_file = open(train_lda, 'r')
        train_lda_file_list = train_lda_file.readlines() 
        lda_names = train_lda_file_list[0].split(",")
        train_lda_file_list.pop(0)
        test_lda_file = open(test_lda,'r')
        test_lda_file_list = test_lda_file.readlines()
        test_lda_file_list.pop(0)
        
        lda_train_rows = []
        for line in train_lda_file_list:
            line = line.replace("\n",'')
            lda_train_rows.append([float(r) for r in line.split(',')])
        train_lda_file.close()
        
        
        lda_test_rows = []
        for line in test_lda_file_list:
            line = line.replace("\n",'')
            lda_test_rows.append([float(r) for r in line.split(',')])
        test_lda_file.close()
            
        if (features or POS or charNgrams or POSgrams):
            train_matrix = np.concatenate([train_matrix, np.matrix(lda_train_rows)], axis = 1)
            test_matrix = np.concatenate([test_matrix, np.matrix(lda_test_rows)], axis = 1)
        else:
            train_matrix = np.matrix(lda_train_rows)
            test_matrix = np.matrix(lda_test_rows)
        names = names + lda_names
    
    if removeCenter:
        
        extreme_indexes = []
        for i in range(0,len(texts_train)):
            if (nominates_train[i] > thresholdPos or nominates_train[i]<thresholdNeg):
                extreme_indexes.append(i)
        if (features or POS or charNgrams or POSgrams or lda):        
            train_matrix = train_matrix[extreme_indexes,:]
        labels_train = [labels_train[i] for i in extreme_indexes]
        texts_train = [texts_train[i] for i in extreme_indexes]
        """
        extreme_indexes = []
        for i in range(0,len(texts_test)):
            if (nominates_test[i] > thresholdPosTest or nominates_test[i]<thresholdNegTest):
                extreme_indexes.append(i)
        if (features or POS or charNgrams or POSgrams or lda):       
            test_matrix = test_matrix[extreme_indexes,:]
        
        texts_test = [texts_test[i] for i in extreme_indexes]
        labels_test = [labels_test[i] for i in extreme_indexes]
        nominates_test = [nominates_test[i] for i in extreme_indexes]
        """
    if BoW:
        print("Generating Bag of Words")
        
        #vocab_f = open(vocab_path, 'r')
        #vocab = vocab_f.readline().split(',')
        vectorizer = CountVectorizer(token_pattern = '[a-zA-Z]+', stop_words='english')
        bow_train = vectorizer.fit_transform(texts_train)
        bow_test = vectorizer.transform(texts_test)
        if (features or POS or charNgrams or POSgrams or lda):
            train_matrix = hstack((bow_train,train_matrix))
            test_matrix = hstack((bow_test,test_matrix))
        else:
            train_matrix = bow_train
            test_matrix = bow_test
        bow_names = vectorizer.get_feature_names()
        names = bow_names + names
    
    if tfidf:
        print("Generating TFIDF")
        
        #vocab_f = open(vocab_path, 'r')
        #vocab = vocab_f.readline().split(',')
        vectorizer = TfidfVectorizer(token_pattern = '[a-zA-Z]+', stop_words='english')
        bow_train = vectorizer.fit_transform(texts_train)
        bow_test = vectorizer.transform(texts_test)
        if (features or POS or charNgrams or POSgrams or BoW):
            train_matrix = hstack((bow_train,train_matrix))
            test_matrix = hstack((bow_test,test_matrix))
        else:
            train_matrix = bow_train
            test_matrix = bow_test
        bow_names = vectorizer.get_feature_names()
        names = bow_names + names
    
    if not BoW or not tfidf:
        train_matrix = sparse.csc_matrix(train_matrix)
        test_matrix = sparse.csc_matrix(test_matrix)
    
    if binary:
        transformer = Binarizer().fit(train_matrix)
        train_matrix = transformer.transform(train_matrix)
        
        transformer = Binarizer().fit(test_matrix)
        test_matrix = transformer.transform(test_matrix)
    

        
        
    print("Training the Naive Bayes classifier")
    clf = MultinomialNB()
    clf.fit(train_matrix, labels_train)
    pred = clf.predict(test_matrix)

    print("Naive Bayes")
    print("Accuracy:  "+str(Eval.Accuracy(labels_test, pred.tolist())))
    print("Precision: "+str(Eval.Precision(labels_test, pred.tolist())))
    print("Recall: "+str(+Eval.Recall(labels_test, pred.tolist())))
    cm = confusion_matrix(labels_test, pred)
    print(cm)
    #print("Speaker accuracy: " + str(Eval.SpeakerAccuracy(112, pred)))
    
    nb_ac = Eval.Accuracy(labels_test, pred.tolist())
    
    Eval.histogram(nominates_test,labels_test,pred.tolist(),10, 'Naive Bayes', 'c')
    
    
    a = clf.feature_log_prob_[0] - clf.feature_log_prob_[1]
    b = [x*y for x,y in  zip(a, train_matrix.mean(axis=0).tolist()[0])]
    coefs_with_fns = sorted(zip(b, names)) 
    top = zip(coefs_with_fns[:20], coefs_with_fns[:-(20 + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_2, fn_2, coef_1, fn_1))
    
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(train_matrix, labels_train)
    pred = clf.predict(test_matrix)
    
    print("Logistic Regression")
    print("Accuracy: "+str(Eval.Accuracy(labels_test, pred.tolist())))
    print("Precision: "+str(Eval.Precision(labels_test, pred.tolist())))
    print("Recall: "+str(Eval.Recall(labels_test, pred.tolist())))
    cm = confusion_matrix(labels_test, pred)
    print(cm)
    #print("Speaker accuracy: " + str(Eval.SpeakerAccuracy(112, pred)))

    Eval.histogram(nominates_test,labels_test,pred.tolist(),10, 'Logistic Regression', 'b')
    
    plt.legend(loc=1, ncol=1)
    
    b = [x*y for x,y in  zip(clf.coef_[0], train_matrix.mean(axis=0).tolist()[0])]
    coefs_with_fns = sorted(zip(b, names))
    top = zip(coefs_with_fns[:20], coefs_with_fns[:-(20 + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
            print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
    
    log_ac = Eval.Accuracy(labels_test, pred.tolist())
    
    #return nb_ac, log_ac, labels_test, labels_train
