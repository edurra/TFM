# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:23:28 2019

@author: Eduardo
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import random

def subSample(labels, texts):
    
    indices_neg = []
    indices_pos = []
    
    random.seed(1)
    indices_shuffled = random.sample(range(0,len(texts)),len(texts))
    texts_shuffled = [texts[i] for i in indices_shuffled]
    labels_shuffled = [labels[i] for i in indices_shuffled]
    
    labels_neg = [l for l in labels if l == -1.0]
    labels_pos = [l for l in labels if l == 1.0]
    
    minClass = min(len(labels_neg),len(labels_pos))
    final_labels = []
    final_texts = []
    nPos = 0
    nNeg = 0
    for i in range(0, len(texts)):
        if(labels_shuffled[i] == -1.0 and nNeg < minClass):
            final_labels.append(-1.0)
            final_texts.append(texts_shuffled[i])
            nNeg += 1
        if(labels_shuffled[i] == 1.0 and nPos < minClass):
            final_labels.append(1.0)
            final_texts.append(texts_shuffled[i])
            nPos += 1
    return final_labels, final_texts

def readDataSet(dataset_path, dw_threshold):
    
    data_file = open(dataset_path, 'r', encoding='latin-1')
    labels = []
    texts = []
    nominates = []
    
    
    for line in data_file:
        line_split = line.split("|")
        nominate = line_split[0]
        
        text = line_split[2]   
        
        if(float(nominate) < dw_threshold):
            labels.append(-1.0)
        else:
            labels.append(1.0)
        """ 
        if(float(nominate) < -0.1):
            labels.append(-1.0)
        elif (float(nominate) >= -0.1 and float(nominate) < 0.1):
            labels.append(0.0)
        else:
           
            labels.append(1.0)
        """
        texts.append(text)
        nominates.append(float(nominate))
    return (labels, texts, nominates)

def readDataSetThreeClasses(dataset_path, dw_threshold):
    
    data_file = open(dataset_path, 'r', encoding='latin-1')
    labels = []
    texts = []
    nominates = []
    
    
    for line in data_file:
        line_split = line.split("|")
        nominate = line_split[0]
        
        text = line_split[2]   
        """
        if(float(nominate) < dw_threshold):
            labels.append(-1.0)
        else:
            labels.append(1.0)
        """ 
        if(float(nominate) < -0.35):
            labels.append(-1.0)
        elif (float(nominate) >= -0.35 and float(nominate) < 0.35):
            labels.append(0.0)
        else:
           
            labels.append(1.0)
        
        texts.append(text)
        nominates.append(float(nominate))
    return (labels, texts, nominates)

def Accuracy(original, pred):
    return np.sum(np.equal(pred, original)) / float(len(original))

def  Recall(original, pred):
               
    positives = []
    for i in range(0, len(original)):
        if original[i] == 1.0:
            positives.append(1)
        else:
            positives.append(0)
    true_positives = []
    for i in range(0, len(original)):
        if original[i] == 1.0 and pred[i] == 1.0:
            true_positives.append(1)
        else:
            true_positives.append(0)
    return np.sum(true_positives)/np.sum(positives)

def  Precision(original, pred):
        
    pred_positives = []
    
    for i in range(0, len(pred)):
        if pred[i] == 1.0:
            pred_positives.append(1)
        else:
            pred_positives.append(0)
    true_positives = []
    for i in range(0, len(original)):
        if original[i] == 1.0 and pred[i] == 1.0:
            true_positives.append(1)
        else:
            true_positives.append(0)
    return np.sum(true_positives)/np.sum(pred_positives)

def histogram(nominates, labels_test, predictions, n, label, color):
    #2*n is the number of segments in which we are divinding the x axis
    
    ncorrect = [0]*2*n
    nsamples = [1]*2*n
    accuracies = [0]*2*n
    
    for i in range(0, len(nominates)):
        #indexes vary from 0 to n and nominates*n from -n to n-1
        index = math.floor(nominates[i]*n) + n
        if(labels_test[i] == predictions[i]):
            ncorrect[index] += 1
        nsamples[index] +=1
    
    for i in range(0, len(ncorrect)):
        accuracies[i] = float(ncorrect[i])/float(nsamples[i])
    
    x_min = math.floor(min(nominates)*n)
    x_max = math.ceil(max(nominates)*n)+1
    ranges = [i/(n) for i in list(range(x_min,x_max))]
    ranges_middle = []
    for i in range(0,len(ranges)-1):
        ranges_middle.append((ranges[i]+ranges[i+1])/2)
    #ranges = [i/(n) for i in list(range(-1*n,n))]
    if(len(accuracies[x_min+n:x_max+n]) > len(ranges_middle)):
        plt.plot(ranges_middle, accuracies[x_min+n:x_max+n-1], label = label, color = color)
    else:
        plt.plot(ranges_middle, accuracies[x_min+n:x_max+n], label = label, color = color)
    plt.xlabel("Nominate")
    plt.ylabel("Accuracy")
    

def SpeakerAccuracy(n, predictions):
    path_speaker = "C:/Users/Eduardo/Desktop/2 cuatri IIT/TFM/Datasets/hein-daily/hein-daily/longTexts/speeches_"+str(n)+"_speakerId.txt"
    texts_path = "C:/Users/Eduardo/Desktop/2 cuatri IIT/TFM/Datasets/hein-daily/hein-daily/longTexts/speeches_"+str(n)+"_dwnominate_nonames.txt"
    labels, texts, nominates = readDataSet(texts_path, 0)
    
    speakers = []
    
    f = open(path_speaker, 'r')
    for line in f.readlines():
        speakerid = int(line.split('|')[1].replace('\n',''))
        speakers.append(speakerid)
    f.close()
    speaker_pos = {}
    speaker_neg = {}
    speaker_labels = {}
    
    for i in range(0,len(speakers)):
        if speakers[i] not in speaker_pos.keys():
            speaker_pos[speakers[i]] = 0
        if speakers[i] not in speaker_neg.keys():
            speaker_neg[speakers[i]] = 0
        if speakers[i] not in speaker_labels.keys():
            speaker_labels[speakers[i]] = labels[i]
    
    for i in range(0,len(predictions)):
        speaker = speakers[i]
        
        if predictions[i] == 1.0:
            speaker_pos[speaker] = speaker_pos[speaker] + 1
        else:
            speaker_neg[speaker] = speaker_neg[speaker] + 1
    nCorrect = 0
    posCorrect = 0
    negCorrect = 0
    for speaker in speaker_labels.keys():
        result = 1.0
        if speaker_pos[speaker] < speaker_neg[speaker]:
            result = -1.0
        
        if speaker_pos[speaker] == speaker_neg[speaker]:
            indices = [i for i in list(range(0,len(labels))) if speakers[i] == speaker]
            index = -1
            maxLength = 0
            
            for i in indices:
                if len(texts[i]) > maxLength:
                    maxLength = len(texts[i])
                    index = i
            result = labels[index]
        
        if result == speaker_labels[speaker]:
            nCorrect += 1
            if speaker_labels[speaker] == 1.0:
                posCorrect += 1
            if speaker_labels[speaker] == -1.0:
                negCorrect += 1
                
    print("Positive correct: "+ str(posCorrect))
    print("Total positive: " + str(len([i for i in speaker_labels.keys() if speaker_labels.get(i) == 1.0])))
    print("Negative correct: " + str(negCorrect))
    print("Total negative: " + str(len([i for i in speaker_labels.keys() if speaker_labels.get(i) == -1.0])))

    
    return float(nCorrect/len(speaker_labels.keys()))

def SpeakerAccuracyThreshold(n, predictions, threshold):
    path_speaker = "C:/Users/Eduardo/Desktop/2 cuatri IIT/TFM/Datasets/hein-daily/hein-daily/longTexts/speeches_"+str(n)+"_speakerId.txt"
    texts_path = "C:/Users/Eduardo/Desktop/2 cuatri IIT/TFM/Datasets/hein-daily/hein-daily/longTexts/speeches_"+str(n)+"_dwnominate_nonames.txt"
    labels, texts, nominates = readDataSet(texts_path, 0)
    
    speakers = []
    
    f = open(path_speaker, 'r')
    i = 0
    for line in f.readlines():
        speakerid = int(line.split('|')[1].replace('\n',''))
        if nominates[i] > threshold[1] or nominates[i] < threshold[0]:
            speakers.append(speakerid)
        i+=1

    f.close()
    speaker_pos = {}
    speaker_neg = {}
    speaker_labels = {}
    
    for i in range(0,len(speakers)):
        if speakers[i] not in speaker_pos.keys():
            speaker_pos[speakers[i]] = 0
        if speakers[i] not in speaker_neg.keys():
            speaker_neg[speakers[i]] = 0
        if speakers[i] not in speaker_labels.keys():
            speaker_labels[speakers[i]] = labels[i]
    
    for i in range(0,len(predictions)):
        speaker = speakers[i]
        
        if predictions[i] == 1.0:
            speaker_pos[speaker] = speaker_pos[speaker] + 1
        else:
            speaker_neg[speaker] = speaker_neg[speaker] + 1
    nCorrect = 0
    posCorrect = 0
    negCorrect = 0
    total = 0
    for speaker in speaker_labels.keys():
        total += 1
        result = 1.0
        if speaker_pos[speaker] < speaker_neg[speaker]:
            result = -1.0
        
        if speaker_pos[speaker] == speaker_neg[speaker]:
            indices = [i for i in list(range(0,len(labels))) if speakers[i] == speaker]
            index = -1
            maxLength = 0
            
            for i in indices:
                if len(texts[i]) > maxLength:
                    maxLength = len(texts[i])
                    index = i
            result = labels[index]
        
        if result == speaker_labels[speaker]:
            nCorrect += 1
            if speaker_labels[speaker] == 1.0:
                posCorrect += 1
            if speaker_labels[speaker] == -1.0:
                negCorrect += 1
                
    print("Positive correct: "+ str(posCorrect))
    print("Total positive: " + str(len([i for i in speaker_labels.keys() if speaker_labels.get(i) == 1.0])))
    print("Negative correct: " + str(negCorrect))
    print("Total negative: " + str(len([i for i in speaker_labels.keys() if speaker_labels.get(i) == -1.0])))
    print("Total: " +str(total))
    
    return float(nCorrect/total)
