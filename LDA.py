# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:01:18 2019

@author: Eduardo
"""
import gensim
import Eval
from sklearn.feature_extraction.text import CountVectorizer
import datetime
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import PorterStemmer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

train_n = 110
test_n = 112
texts_train_path = "dir/speeches_"+str(train_n)+"_dwnominate_nonames.txt"
texts_test_path = "dir/speeches_"+str(test_n)+"_dwnominate_nonames.txt"

labels_train, texts_train, nominates_train = Eval.readDataSet(texts_train_path, 0)
labels_test, texts_test, nominates_test = Eval.readDataSet(texts_test_path, 0)

def preprocess(text, stop, stemmer):
    
    result = []
    for w in text.split(" "):
        if len(w) > 3 and not stop.get(w):
            result.append(stemmer.stem(w.lower()))
    return result

def saveHTML(lda_model, path, dictionary, corpus):
    import pyLDAvis.gensim
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics = False)

    pyLDAvis.save_html(vis, path)

def initialize(texts_train, texts_test):    
    v = CountVectorizer(stop_words="english")
    s = v.get_stop_words()
    stop_w = [x for x in s]
    stop = {}
    
    for w in stop_w:
        stop[w] = 1
    
    print("Preprocessing training file")
    ps = PorterStemmer()
    texts_train_p = [preprocess(texts_train[i], stop, ps) for i in list(range(0,len(texts_train)))]
    
    print("Preprocessing test file")
    texts_test_p = [preprocess(texts_test[i], stop, ps) for i in list(range(0,len(texts_test)))]
    
    print("Generating dictionary")
    dictionary = gensim.corpora.Dictionary(texts_train_p)
    dictionary.filter_extremes(no_below=15, no_above=0.5)
    
    bow_corpus = [dictionary.doc2bow(doc) for doc in texts_train_p]
    test_corpus = [dictionary.doc2bow(doc) for doc in texts_test_p]
    
    return dictionary, bow_corpus, test_corpus

def lda(bow_corpus, n_topics, dictionary, showWords = False, saveModel = False, returnModel = True):
    print("Generating LDA model")
    t = datetime.datetime.now()
    lda_model = gensim.models.LdaModel(bow_corpus, num_topics=n_topics, id2word=dictionary)
    
    print(str((datetime.datetime.now() - t).seconds/60) + " minutes")
    
    if showWords:
        for idx, topic in lda_model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))
    if saveModel:
        lda_model.save('lda/lda_model_'+str(n_topics))
    return lda_model
    
def getBestN(test_corpus, n_topics, metric):
    if metric == 'perplexity':
        perplexities = []
        for n in n_topics:
            print(n)
            model = gensim.models.LdaModel.load('lda/lda_model_'+str(n))
            perplexities.append(2**(-1*model.log_perplexity(test_corpus)))
        plt.plot(n_topics, perplexities)
        plt.xlabel("Number of topics")
        plt.ylabel("Perplexity")
    
    if metric == 'coherence':
        coherences = []
        for n in n_topics:
            print(n)
            model = gensim.models.LdaModel.load('lda/lda_model_'+str(n))
            cm = CoherenceModel(model=model, corpus=test_corpus, coherence='c_v')
            coherences.append(cm.get_coherence())
        plt.plot(n_topics, coherences)
        plt.xlabel("Number of topics")
        plt.ylabel("Coherence")

def getCorpusTopics(corpus, model, labels):
    result = []
    for t in corpus:
        topic = sorted(model[t], key = lambda x: float(x[1]), reverse = True)[0][0]
        result.append(topic)
    left_topics = []
    right_topics = []
    for i in range(0, len(labels)):
        if labels[i] == 1.0:
            right_topics.append(result[i])
        else:
            left_topics.append(result[i])
    return right_topics, left_topics

def plotTopics(right, left, numberOfTopics, normalize = True):
    count_r = {}
    count_l = {}
    for i in range(0, numberOfTopics):
        count_r[i] = 0
        count_l[i] = 0
    for t in right:
        count_r[t] += 1
    for t in left:
        count_l[t] += 1
    if normalize:
        for i in range(0, numberOfTopics):
            count_r[i] = count_r[i] / len(right)
            count_l[i] = count_l[i] / len(left)
        
    rAxis = []
    lAxis = []
    for i in range(0, numberOfTopics):
        rAxis.append(i + 1 + 0.15)
        lAxis.append(i + 1 - 0.15)
    plt.bar(rAxis, count_r.values(), edgecolor = 'black', width = 0.3, color = 'C1', label = "Right side")
    plt.bar(lAxis, count_l.values(), edgecolor = 'black', width = 0.3, color = 'C2', label = "Left side")
    plt.legend(loc='best')
    plt.xlabel("Topic")
    plt.ylabel("Number of texts")
    plt.xticks(list(range(1,numberOfTopics+1)))
    
    if normalize:
        plt.ylabel("Percentage of texts")

def plotTopicsNoClass(right, left, numberOfTopics, normalize = True):
    count = {}
    for i in range(0, numberOfTopics):
        count[i] = 0     
    for t in right:
        count[t] += 1
    for t in left:
        count[t] += 1
    if normalize:
        for i in range(0, numberOfTopics):
            count[i] = count[i] / (len(right)+len(left))
        
    rAxis = []
    for i in range(1, numberOfTopics+1):
        rAxis.append(i)
    plt.bar(rAxis, count.values(), edgecolor = 'black', width = 0.5)
    plt.xlabel("Topic")
    plt.ylabel("Number of texts")
    plt.xticks(list(range(1,numberOfTopics+1)))
    
    if normalize:
        plt.ylabel("Percentage of texts")
        
def sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sent = analyzer.polarity_scores(text)
    
    label = 1.0
    
    pos = sent['pos']
    neg = sent['neg']
    neu = sent['neu']
    
    if neg >= pos:
        label = -1.0
    """
    if neu > neg and neu > pos:
        label = 0.0
    """
    return label

def readSentiment():
    t = open('sentiment/polarity_train.txt', 'r')
    te = open('sentiment/polarity_test.txt', 'r')
    
    train = []
    test = []
    
    for line in t.readlines():
        train.append(float(line.replace('\n', '')))
    
    for line in te.readlines():
        test.append(float(line.replace('\n', '')))
    
    return train, test

def sentimentRightLeft(labels, sentiment):
    left = []
    right = []
    for i in range(0, len(labels)):
        if labels[i] == 1.0:
            right.append(sentiment[i])
        else:
            left.append(sentiment[i])
    return left, right

def plotPolarity(left_sent, right_sent, left_topics, right_topics, numberOfTopics, labels):
    count_pos_left = {}
    count_neg_left = {}
    
    count_pos_right = {}
    count_neg_right = {}
    
    for i in range(0, numberOfTopics):
        count_pos_left[i] = 0
        count_neg_left[i] = 0
        
        count_pos_right[i] = 0
        count_neg_right[i] = 0
    
    for i in range(0, len(left_sent)):
        if left_sent[i] == 1.0:
            count_pos_left[left_topics[i]] +=1
        else:
            count_neg_left[left_topics[i]] += 1
    for i in range(0, len(right_sent)):
        if right_sent[i] == 1.0:
            count_pos_right[right_topics[i]] +=1
        else:
            count_neg_right[right_topics[i]] += 1
    
    for i in range(0, numberOfTopics):
        count_pos_left[i] = count_pos_left[i] / len(left_topics)
        count_neg_left[i] = count_neg_left[i] / len(left_topics)
        count_pos_right[i] = count_pos_right[i] / len(right_topics)
        count_neg_right[i] = count_neg_right[i] / len(right_topics)
    
    """
    yaxis1 = np.array(list(count_pos_left.values()))
    yaxis2 = np.array(list(count_neg_left.values()))
    xaxis = list(range(0,14))          
    plt.bar(xaxis, yaxis1, edgecolor = 'black', label = 'Positive attitude')
    plt.bar(xaxis, yaxis2, bottom=yaxis1, color = 'red', edgecolor='black', label= 'Negative attitude')
    plt.xlabel('Topic')
    plt.ylabel('Percentage of texts')
    plt.ylim(0,0.19)
    plt.legend(loc='best')
    plt.show()
    
    yaxis1 = np.array(list(count_pos_right.values()))
    yaxis2 = np.array(list(count_neg_right.values()))
    xaxis = list(range(0,14))          
    plt.bar(xaxis, yaxis1, edgecolor = 'black', label = 'Positive attitude')
    plt.bar(xaxis, yaxis2, bottom=yaxis1, color = 'red', edgecolor='black', label= 'Negative attitude')
    plt.xlabel('Topic')
    plt.ylabel('Percentage of texts')
    plt.legend(loc='best')
    plt.ylim(0,0.19)
    """
    rAxis = []
    lAxis = []
    for i in range(0, numberOfTopics):
        rAxis.append(i + 1 + 0.15)
        lAxis.append(i + 1 - 0.15)
    yaxis = []
    for i, j in zip(list(count_pos_left.values()), list(count_neg_left.values())):
        yaxis.append(i/j)
    
    xaxis = list(range(0,14))          
    plt.bar(lAxis, yaxis, edgecolor = 'black', label = 'Left side', width = 0.3)
    yaxis = []
    for i, j in zip(list(count_pos_right.values()), list(count_neg_right.values())):
        yaxis.append(i/j)
    xaxis = list(range(0,14))          
    plt.bar(rAxis, yaxis, edgecolor = 'black', label = 'Right side', width = 0.3)
    plt.xlabel('Topic')
    plt.ylabel('Positive/negative ratio')
    plt.legend(loc='best')
    plt.xticks(list(range(1,numberOfTopics+1)))
    plt.show()

def extractFeatures(corpus, model, path, nTopics):
    f = open(path, 'w')
    nTopicsList = list(range(0, nTopics))
    names = ["Topic " + str(i) for i in nTopicsList]
    f.write(",".join(names) + "\n")
    count = 0
    for t in corpus:
        if count%1000 == 0:
            print(count)
        count += 1
        values = {}
        for i in nTopicsList:
            values[i] = 0
        for j in model[t]:
            values[j[0]] = j[1]
        f.write(",".join([str(i) for i in values.values()]) + "\n")
    f.close()

def wordsInTopics(lda_model, dictionary, nTopics):
    words = {}
    for i in range(0, nTopics):
        terms = lda_model.get_topic_terms(i, 30)
        for t in terms:
            words[dictionary[t[0]]] = 1
    return words

def preprocessRemovingWords(text, stop, stemmer, wordsToRemove):
    
    result = []
    for w in text.split(" "):
        if len(w) > 3 and not stop.get(w):
            w_stemmed = stemmer.stem(w.lower())
            if not wordsToRemove.get(w_stemmed):
                result.append(w_stemmed)
    return result

def initializeRemovingWords(texts_train, texts_test, wordsToRemove):    
    v = CountVectorizer(stop_words="english")
    s = v.get_stop_words()
    stop_w = [x for x in s]
    stop = {}
    
    for w in stop_w:
        stop[w] = 1
    
    print("Preprocessing training file")
    ps = PorterStemmer()
    texts_train_p = [preprocessRemovingWords(texts_train[i], stop, ps, wordsToRemove) for i in list(range(0,len(texts_train)))]
    
    print("Preprocessing test file")
    texts_test_p = [preprocessRemovingWords(texts_test[i], stop, ps, wordsToRemove) for i in list(range(0,len(texts_test)))]
    
    print("Generating dictionary")
    dictionary = gensim.corpora.Dictionary(texts_train_p)
    dictionary.filter_extremes(no_below=15, no_above=0.5)
    
    bow_corpus = [dictionary.doc2bow(doc) for doc in texts_train_p]
    test_corpus = [dictionary.doc2bow(doc) for doc in texts_test_p]
    
    return dictionary, bow_corpus, test_corpus