# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:12:57 2019

@author: Eduardo
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
import math
import nltk
from scipy.sparse import find
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import Eval
from sklearn.feature_selection import mutual_info_classif
import operator

train_n = 110
test_n = 112

texts_train_path = "directory/speeches_"+str(train_n)+"_dwnominate_nonames.txt"
texts_test_path = "directory/speeches_"+str(test_n)+"_dwnominate_nonames.txt"

path_speaker_train = "directory/speeches_"+str(train_n)+"_speakerId.txt"
path_speaker_test = "directory/speeches_"+str(test_n)+"_speakerId.txt"

ig_path = "directory/informationGain.txt"


labels_train, texts_train, nominates_train = Eval.readDataSet(texts_train_path, 0)
labels_test, texts_test, nominates_test = Eval.readDataSet(texts_test_path, 0)


def entropy(labels):
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in freqdist]
    return -sum(p * math.log(p,2) for p in probs)

def informationGain(texts, labels, nFeatures = 10000):
    vectorizer = CountVectorizer(token_pattern = '[a-zA-Z]+', stop_words='english')
    bow = vectorizer.fit_transform(texts)
    transformer = Binarizer().fit(bow)
    bow = transformer.transform(bow)
    names = vectorizer.get_feature_names()
    
    if nFeatures != -1:
        pos_train = []
        neg_train = []
        for i in range(0,len(labels_train)):
            if labels_train[i] == -1.0:
                neg_train.append(i)
            else:
                pos_train.append(i)
                
        pos_matrix = bow.tocsr()[pos_train,:]
        neg_matrix = bow.tocsr()[neg_train,:]
        diff = [abs(x - y) for x,y in zip(pos_matrix.mean(axis = 0).tolist()[0], neg_matrix.mean(axis = 0).tolist()[0])]
       
        indexes = []
        
        indexes_sorted = [i[0] for i in sorted(enumerate(diff), key=lambda x:x[1])]
        names_sorted = [names[i] for i in indexes_sorted]
        
        indexes = indexes_sorted[len(indexes_sorted)-nFeatures:len(indexes_sorted)]
        names = names_sorted[len(indexes_sorted)-nFeatures:len(indexes_sorted)]
        bow = bow.tocsr()[:,indexes]
    
    info_gain = {}
    
    labels_entropy = entropy(labels)
    count = 0
    for w in names:
        count += 1
        if count%500 == 0:
            print(count/bow.shape[1]*100)
        texts_with_w_labels = []
        texts_without_w_labels = []
        index = names.index(w)
        column = bow[:,index]
        
        with_indices = find(column)[0].tolist()
        texts_with_w_labels = [labels[i] for i in list(range(0,len(labels))) if i in with_indices ]
        texts_without_w_labels = [labels[i] for i in list(range(0,len(labels))) if i not in with_indices ]
        info_gain_w = labels_entropy - (float(len(texts_with_w_labels))/float(len(labels))) * entropy(texts_with_w_labels) -(float(len(texts_without_w_labels))/float(len(labels))) * entropy(texts_without_w_labels)
    
        info_gain[w] = info_gain_w
        
    return info_gain

def InformationGainMutualInfo(texts, labels, nFeatures = 10000):
    if nFeatures == -1:
        vectorizer = CountVectorizer(token_pattern = '[a-zA-Z]+', stop_words='english')
    else:
        vectorizer = CountVectorizer(token_pattern = '[a-zA-Z]+', stop_words='english', max_features = nFeatures)
    bow = vectorizer.fit_transform(texts)
    res = dict(zip(vectorizer.get_feature_names(),
               mutual_info_classif(bow, labels_train, discrete_features=True)
               ))
    return res

def IGFromTexts(texts, info_gain):
    tokenizer = RegexpTokenizer(r'\w+')
    info_gain_texts = {}
    for i in range(0,len(texts)):
        if i%10000 == 0:
            print(i/len(texts)*100)
        words = tokenizer.tokenize(texts[i])
        avg_IG = 0
        n = 0
        for w in words:
            w = w.lower()
            if info_gain.get(w):
                avg_IG += info_gain[w]
                n += 1
        avg_IG = avg_IG/n
        info_gain_texts[i] = avg_IG
    
    return info_gain_texts

def PlotIG(info_gain_texts, label = None):
    
    if label!=None:
        plt.hist(info_gain_texts.values(), bins = 50, label = label)
        plt.legend(loc='best')
    else:
        plt.hist(info_gain_texts.values(), bins = 50)
    
    plt.xlabel('Information gain')
    plt.ylabel('Number of texts')
    
def AvgTextLength(texts):
    lengths = {}
    tokenizer = RegexpTokenizer(r'\w+')
    
    for i in range(0, len(texts)):
        if i%10000 == 0:
            print(i/len(texts)*100)
        words = tokenizer.tokenize(texts[i])
        lengths[i] = len(words)
    return lengths

def AvgTextLengthCharacters(texts):
    lengths = {}    
    for i in range(0, len(texts)):
        if i%10000 == 0:
            print(i/len(texts)*100)
        lengths[i] = len(texts[i])
    return lengths

def PlotLenghts(lengths):
    plt.hist(lengths.values(), bins = 500)
    plt.xlim(0, 3000)
    plt.xlabel('Length (words)')
    plt.ylabel('Number of texts')

def PlotLengthsIG(lengths, info_gain_texts):
    xAxis = [i*100 for i in list(range(1,25))]
    values = {}
    samples = {}
    ratio = {}
    for x in xAxis:
        values[x] = 0
        samples[x] = 1
        ratio[x] = 0
    for i in range(0,len(lengths.keys())):
        length = lengths[i]
        if length >= 100 and length <= 2400:
            l = length/100
            l_rounded = round(l, 0)
            l = l_rounded*100
            samples[l] = samples[l] + 1
            values[l] = values[l] + info_gain_texts[i]
    for x in xAxis:
        if(samples[x] != 0):
            ratio[x] = values[x] / samples[x]
    plt.plot(xAxis, ratio.values())
    plt.xlim(200, max(xAxis))
    plt.ylim(0.00085, 0.001)
    plt.xticks(xAxis, rotation = 'vertical')
    plt.xlabel("Text length (words)")
    plt.ylabel("Information gain")

def readIGFile(path):
    f = open(path, 'r')
    f_l = f.readlines()
    info_gain = {}
    for line in f_l:
        l_s = line.split(",")
        info_gain[l_s[0]] = float(l_s[1].replace("\n",''))
    f.close()
    return info_gain

def PosNeg(information_gain_texts, lengths, labels):
    pos_ig = {}
    neg_ig = {}
    pos_lengths = {}
    neg_lengths = {}
    for i in range(0, len(labels)):
        if labels[i] == 1.0:
            pos_ig[i] = information_gain_texts[i]
            pos_lengths[i] = lengths[i]
        else:
            neg_ig[i] = information_gain_texts[i]
            neg_lengths[i] = lengths[i]
    return pos_ig, neg_ig, pos_lengths, neg_lengths

def getSpeakerTexts(path_speaker):
    speakers = []
    
    f = open(path_speaker, 'r')
    for line in f.readlines():
        speakerid = int(line.split('|')[1].replace('\n',''))
        speakers.append(speakerid)
    f.close()
    speaker_allTexts = {}
    
    for i in range(0,len(speakers)):
        
        if speakers[i] not in speaker_allTexts.keys():
            speaker_allTexts[speakers[i]] = [i]
        else:
            speaker_allTexts[speakers[i]] = speaker_allTexts[speakers[i]] + [i]
            
    
    return speaker_allTexts

def getSpeakersIG(speaker_allTexts, info_gain_texts):
    info_gain_speakers = {}
    for k in speaker_allTexts.keys():
        texts = speaker_allTexts[k]
        ig = sum([info_gain_texts[i] for i in texts])/len(texts)
        info_gain_speakers[k] = ig
    
    return info_gain_speakers

def PlotIGSpeakers(info_gain_speakers):
    plt.hist(info_gain_speakers.values(), bins = 40)
    plt.xlabel('Information gain')
    plt.ylabel('Number of speakers')
    plt.xticks(rotation = 'vertical')
    
def SpeakerDistrubution(path_speaker, labels):

    speakers = []
    
    f = open(path_speaker, 'r')
    for line in f.readlines():
        speakerid = int(line.split('|')[1].replace('\n',''))
        speakers.append(speakerid)
    f.close()
    speaker_labels = {}
    speaker_texts = {} 
    for i in range(0,len(speakers)):
        
        if speakers[i] not in speaker_labels.keys():
            speaker_labels[speakers[i]] = labels[i]
            speaker_texts[speakers[i]] = 1
        else:
            speaker_texts[speakers[i]] = speaker_texts[speakers[i]] + 1
    
    return speaker_labels, speaker_texts

def PlotSpeakerTexts(speaker_texts):
    xAxis = [i*10 for i in list(range(0,100))]
    speakersNumber = {}
    for x in xAxis:
        speakersNumber[x] = 0
    for k in speaker_texts.keys():
        n_texts = speaker_texts[k]
        l = n_texts/10
        l_rounded = math.floor(l)
        l = l_rounded*10
        speakersNumber[l] += 1    
    
    plt.plot(xAxis, speakersNumber.values())
    plt.xlim(min(xAxis),200)
    plt.xlabel("Number of texts")
    plt.ylabel("Number of speakers")

def PlotPredictedIG(predictions, labels, info_gain_texts):
    right_pred = {}
    wrong_pred = {}
    avgCorrect = 0
    avgWrong = 0
    for i in range(0,len(labels)):
        if predictions[i] == labels[i]:
            right_pred[i] = info_gain_texts[i]
            avgCorrect += info_gain_texts[i]
        else:
            wrong_pred[i] = info_gain_texts[i]
            avgWrong += info_gain_texts[i]
    avgCorrect = avgCorrect/len(right_pred.keys())
    avgWrong = avgWrong/len(wrong_pred.keys())
    
    print("Average IG in correct predicitons: " + str(avgCorrect))
    print("Average IG in wrong predictons: " + str(avgWrong))
    
    PlotIG(info_gain_texts, label = "All texts")
    PlotIG(right_pred, label = "Correct predictions")
    PlotIG(wrong_pred, label = "Wrong predictions")

def TopWords(info_gain):
    sorted_x = sorted(info_gain.items(), key=operator.itemgetter(1))
    sorted_x.reverse()
    
    for i in range(0,20):
    
        print(sorted_x[i])

def PlotNominateIG(nominates, info_gain_texts):
    xAxis = [i/10 for i in list(range(-10, 10))]
    avgIG = {}
    nsamples = {}
    for x in xAxis:
        avgIG[x] = 0
        nsamples[x] = 0
    for i in range(0, len(nominates)):
        nom = nominates[i]
        nom_rounded = math.floor(nom*10)/10
        avgIG[nom_rounded] = avgIG[nom_rounded] + info_gain_texts[i]
        nsamples[nom_rounded] = nsamples[nom_rounded] + 1
    
    for x in xAxis:
        if nsamples[x] != 0:
            avgIG[x] = avgIG[x]/nsamples[x]
    
    plt.bar(xAxis, avgIG.values(), edgecolor = 'black', width = 0.1)
    plt.xlabel('Nominate')
    plt.ylabel('Information gain')
    plt.xlim(-0.75, 1.0)