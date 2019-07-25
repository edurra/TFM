# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:45:24 2019

@author: Eduardo
"""
from nltk.tokenize import RegexpTokenizer

n = 112
texts_path = "directory/speeches_"+str(n)+"_dwnominate_nonames.txt"
new_file_path = "directory/speeches_"+str(n)+"_dwnominate_nonames_conjunction.txt"

#Elaboration
apposition = ['that is', 'in other words', 'for example']
clarification = ['rather', 'in any case', 'specifically']

#Extension
additive = ['and', 'or', 'moreover']
adversative = ['but', 'yet', 'on the other hand']
verifying = ['besides', 'instead', 'or', 'alternately']

#Enhancement
matter = ['with regards to', 'in one respect', 'otherwise']
#Enhancement-spatiotemporal
simple = ['then', 'next', 'afterwards']
cplx = ['soon', 'meanwhile', 'until', 'now']

manner = ['similarly', 'in a different way', 'likewise']
#Enhancement casual conditional
casual = ['therefore', 'consequently', 'since']
conditional = ['then', 'albeit', 'not with standing']

feature_names = ['Spatiotemporal/Simple', 'Spatiotemporal/Complex', 'CasualConditional/Casual', 'CasualConditional/Conditional', 'Elaboration/Apposition', 'Elaboration/Clarification' 
                 , 'Extension/Additive', 'Extension/Adversative', 'Extension/Verifying', 'Enhancement/Matter', 'Enhancement/Spatiotemporal'
                 , 'Enhancement/Manner', 'Enhancement/CasualConditional', 'Conjunction/Enhancement', 'Conjunction/Extension', 'Conjunction/Elaboration']

data_file = open(texts_path, 'r', encoding='latin-1')
tokenizer = RegexpTokenizer(r'\w+')
new_file = open(new_file_path, 'w', encoding = 'latin-1')
new_file.write(",".join(feature_names)+"\n")
counter = 0
for line in data_file:
    counter += 1
    if(counter%2000 == 0):
        print(counter)
    line_split = line.split("|")
    text = line_split[2]   
    
    features_count = {}
    words = tokenizer.tokenize(text)
    
    for f in feature_names:
        features_count[f] = 0
    for i in range(0, len(words)):
        words2 = '';
        words3 = '';
        words4 = '';
        
        if (i+1 < len(words)):
            words2 = words[i] + " " + words[i+1]
            words2 = words2.lower()
        if (i+2 < len(words)):
            words3 = words[i] + " " + words[i+1] + " " + words[i+2]
            words3 = words3.lower()
        if (i+3 < len(words)):
            words4 = words[i] + " " + words[i+1] + " " + words[i+2] + words[i+3]
            words4 = words4.lower()
            
        if (words[i].lower() in clarification) or (words2 in clarification) or (words3 in clarification) or (words4 in clarification):
            features_count['Elaboration/Clarification'] = features_count['Elaboration/Clarification'] + 1
        if (words[i].lower() in apposition) or (words2 in apposition) or (words3 in apposition) or (words4 in apposition):
            features_count['Elaboration/Apposition'] = features_count['Elaboration/Apposition'] + 1
        if (words[i].lower() in additive) or (words2 in additive) or (words3 in additive) or (words4 in additive):
            features_count['Extension/Additive']  = features_count['Extension/Additive'] + 1
        if (words[i].lower() in adversative) or (words2 in adversative) or (words3 in adversative) or (words4 in adversative):
            features_count['Extension/Adversative']  = features_count['Extension/Adversative'] + 1
        if (words[i].lower() in verifying) or (words2 in verifying) or (words3 in verifying) or (words4 in verifying):
            features_count['Extension/Verifying']  = features_count['Extension/Verifying'] + 1
        if (words[i].lower() in matter) or (words2 in matter) or (words3 in matter) or (words4 in matter):
            features_count['Enhancement/Matter']  = features_count['Enhancement/Matter'] + 1
        if (words[i].lower() in simple) or (words2 in simple) or (words3 in simple) or (words4 in simple):
            features_count['Spatiotemporal/Simple']  = features_count['Spatiotemporal/Simple'] + 1
        if (words[i].lower() in cplx) or (words2 in cplx) or (words3 in cplx) or (words4 in cplx):
            features_count['Spatiotemporal/Complex']  = features_count['Spatiotemporal/Complex'] + 1
        if (words[i].lower() in manner) or (words2 in manner) or (words3 in manner) or (words4 in manner):
            features_count['Enhancement/Manner']  = features_count['Enhancement/Manner'] + 1
        if (words[i].lower() in casual) or (words2 in casual) or (words3 in casual) or (words4 in casual):
            features_count['CasualConditional/Casual']  = features_count['CasualConditional/Casual'] + 1
        if (words[i].lower() in conditional) or (words2 in conditional) or (words3 in conditional) or (words4 in conditional):
            features_count['CasualConditional/Conditional']  = features_count['CasualConditional/Conditional'] + 1
    
        features_count['Enhancement/CasualConditional'] = features_count['CasualConditional/Casual'] + features_count['CasualConditional/Conditional']
        features_count['Enhancement/Spatiotemporal'] = features_count['Spatiotemporal/Complex'] + features_count['Spatiotemporal/Simple']
        
        features_count['Conjunction/Enhancement'] = features_count['Enhancement/CasualConditional'] + features_count['Enhancement/Spatiotemporal'] + features_count['Enhancement/Matter'] + features_count['Enhancement/Manner']
        features_count['Conjunction/Extension'] = features_count['Extension/Additive'] + features_count['Extension/Adversative'] + features_count['Extension/Verifying']
        features_count['Conjunction/Elaboration'] = features_count['Elaboration/Clarification'] + features_count['Elaboration/Apposition']
        
        values = []
        
    for k in feature_names:
        values.append(str(features_count[k]))
    new_file.write(",".join(values)+"\n")

new_file.close()
data_file.close()