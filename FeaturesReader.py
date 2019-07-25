# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:48:28 2019

@author: Eduardo
"""

def readFeatures(new_train, new_test):
    train_file = open(new_train, 'r')
    test_file = open(new_test, 'r')
    
    train = []
    test = []
    
    test_nominates = []
    
    train_labels = []
    test_labels = []
    
    lines = train_file.readlines()
    lines = [l.replace("\n",'') for l in lines]
    line0 = lines[0].split(",")
    names = line0[2:len(line0)]
    lines.pop(0)
    for line in lines:
        
        line_split = line.split(",")
        nominate = float(line_split[0])
        line_split = [float(x) for x in line_split]
        train.append(line_split[2:len(line_split)])
        if(nominate < 0):
            train_labels.append(-1.0)
        else:
            train_labels.append(1.0)
    
    lines = test_file.readlines()
    lines = [l.replace("\n",'') for l in lines]
    
    lines.pop(0)
    
    for line in lines:
        
        line_split = line.split(",")
        nominate = float(line_split[0])
        line_split = [float(x) for x in line_split]
        test.append(line_split[2:len(line_split)])
        if(nominate < 0):
            test_labels.append(-1.0)
        else:
            test_labels.append(1.0)
        test_nominates.append(nominate)
            
    train_file.close()
    test_file.close() 
    
    return train, test, train_labels, test_labels, names