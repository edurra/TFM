# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:35:04 2019

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Binarizer
from scipy import sparse


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

def TestModel(new_train, new_test, texts_train_path, texts_test_path, train_pos, test_pos, new_train_ngrams, new_test_ngrams, nFeaturesList, subsample=False, removeCenter=True, BoW = True, charNgrams = False, POS = False, features = False, POSgrams = False, tfidf = False, binary = False, statistics = False):
    
    names = []
    train, test, train_labels, test_labels, feature_names = FeaturesReader.readFeatures(new_train, new_test) 
    
    labels_train, texts_train, nominates_train = Eval.readDataSet(texts_train_path, 0)
    labels_test, texts_test, nominates_test = Eval.readDataSet(texts_test_path, 0)
    
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
        
 
    if BoW:
        print("Generating Bag of Words")
        
        #vocab_f = open(vocab_path, 'r')
        #vocab = vocab_f.readline().split(',')
        vectorizer = CountVectorizer(token_pattern = '[a-zA-Z]+', stop_words='english')
        bow_train = vectorizer.fit_transform(texts_train)
        bow_test = vectorizer.transform(texts_test)
        if (features or POS or charNgrams or POSgrams):
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
        vectorizer = TfidfVectorizer()
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
    
    if removeCenter:
        extreme_indexes = []
        for i in range(0,len(texts_train)):
            if (nominates_train[i] > 0.2 or nominates_train[i]<-0.2):
                extreme_indexes.append(i)
        train_matrix = train_matrix.tocsr()[extreme_indexes,:]
        labels_train = [labels_train[i] for i in extreme_indexes]
    
    pos_train = []
    neg_train = []
    for i in range(0,len(labels_train)):
        if labels_train[i] == -1.0:
            neg_train.append(i)
        else:
            pos_train.append(i)
        
    pos_matrix = train_matrix.tocsr()[pos_train,:]
    neg_matrix = train_matrix.tocsr()[neg_train,:]
    diff = [abs(x - y) for x,y in zip(pos_matrix.mean(axis = 0).tolist()[0], neg_matrix.mean(axis = 0).tolist()[0])]
   
    indexes = []
    
    indexes_sorted = [i[0] for i in sorted(enumerate(diff), key=lambda x:x[1])]
    names_sorted = [names[i] for i in indexes_sorted]
    
    ac_nb_list = []
    ac_log_list = []
    
    train_matrix_original= train_matrix
    test_matrix_original = test_matrix
    
    for nFeatures in nFeaturesList:
        indexes = indexes_sorted[len(indexes_sorted)-nFeatures:len(indexes_sorted)]
        names = names_sorted[len(indexes_sorted)-nFeatures:len(indexes_sorted)]
        train_matrix = train_matrix_original.tocsr()[:,indexes]
        test_matrix = test_matrix_original.tocsr()[:,indexes]
                                        
        print("Training the Naive Bayes classifier")
        clf = MultinomialNB()
        clf.fit(train_matrix, labels_train)
        pred = clf.predict(test_matrix)
    
        print("Naive Bayes")
        print("Accuracy:  "+str(Eval.Accuracy(labels_test, pred.tolist())))
        print("Precision: "+str(Eval.Precision(labels_test, pred.tolist())))
        print("Recall: "+str(+Eval.Recall(labels_test, pred.tolist())))
        ac_nb = Eval.Accuracy(labels_test, pred.tolist())
        ac_nb_list.append(float(ac_nb))
        
        if statistics:
            Eval.histogram(nominates_test,labels_test,pred.tolist(),10, 'Naive Bayes', 'blue')
            
            a = clf.feature_log_prob_[0] - clf.feature_log_prob_[1]
            b = [x*y for x,y in  zip(a, train_matrix.mean(axis=0).tolist()[0])]
            coefs_with_fns = sorted(zip(b, names)) 
            top = zip(coefs_with_fns[:20], coefs_with_fns[:-(20 + 1):-1])
            for (coef_1, fn_1), (coef_2, fn_2) in top:
                print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_2, fn_2, coef_1, fn_1))
        
        clf = LogisticRegression(solver='saga', max_iter = 2000)
        clf.fit(train_matrix, labels_train)
        pred = clf.predict(test_matrix)
        
        print("Logistic Regression")
        print("Accuracy: "+str(Eval.Accuracy(labels_test, pred.tolist())))
        print("Precision: "+str(Eval.Precision(labels_test, pred.tolist())))
        print("Recall: "+str(Eval.Recall(labels_test, pred.tolist())))
        ac_log = Eval.Accuracy(labels_test, pred.tolist())
        ac_log_list.append(float(ac_log))
        
        if statistics:
            Eval.histogram(nominates_test,labels_test,pred.tolist(),10, 'Logistic Regression', 'orange')
            
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            
            b = [x*y for x,y in  zip(clf.coef_[0], train_matrix.mean(axis=0).tolist()[0])]
            coefs_with_fns = sorted(zip(b, names))
            top = zip(coefs_with_fns[:20], coefs_with_fns[:-(20 + 1):-1])
            for (coef_1, fn_1), (coef_2, fn_2) in top:
                    print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
    
    return ac_nb_list, ac_log_list