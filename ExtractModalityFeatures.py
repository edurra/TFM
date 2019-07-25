# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:46:58 2019

@author: Eduardo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:45:24 2019

@author: Eduardo
"""
from nltk.tokenize import RegexpTokenizer

n = 112
texts_path = "directory/speeches_"+str(n)+"_dwnominate_nonames.txt"
new_file_path = "directory/speeches_"+str(n)+"_dwnominate_nonames_modality.txt"

probability = ['maybe', 'likely', 'probabily']
usuality = ['usually', 'allways', 'sometimes']

readiness = ['will', 'might', 'is ready to']
obligation = ['will', 'ought to', 'should', 'must']

median = ['probably', 'usually']

high = ['definitely', 'always']
low = ['might', 'occasionally']

objective = ['will', 'probably']
subjective = ['should', 'ought to']

implicit = ['perhaps', 'should', 'can']
explict = ['I think', 'it seems', 'it may appear']

feature_names = ['Modalisation/Probability', 'Modalisation/Usuality', 'Modulation/Readiness', 'Modulation/Obligation', 'Type/Modalisation', 'Type/Modulation',
                 'Value/Median', 'Value/Outer', 'Outer/High', 'Outer/Low', 'Orientation/Objective', 'Orientation/Subjective',
                 'Manifestation/Implicit', 'Manifestation/Explicit', 'Modality/Type', 'Modality/Value', 'Modality/Orientation', 
                 'Modality/Manifestation']

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
            
        if (words[i].lower() in probability) or (words2 in probability) or (words3 in probability) or (words4 in probability):
            features_count['Modalisation/Probability'] = features_count['Modalisation/Probability'] + 1
        if (words[i].lower() in usuality) or (words2 in usuality) or (words3 in usuality) or (words4 in usuality):
            features_count['Modalisation/Usuality'] = features_count['Modalisation/Usuality'] + 1
        if (words[i].lower() in readiness) or (words2 in readiness) or (words3 in readiness) or (words4 in readiness):
            features_count['Modulation/Readiness']  = features_count['Modulation/Readiness'] + 1
        if (words[i].lower() in obligation) or (words2 in obligation) or (words3 in obligation) or (words4 in obligation):
            features_count['Modulation/Obligation']  = features_count['Modulation/Obligation'] + 1
        if (words[i].lower() in median) or (words2 in median) or (words3 in median) or (words4 in median):
            features_count['Value/Median']  = features_count['Value/Median'] + 1
        if (words[i].lower() in high) or (words2 in high) or (words3 in high) or (words4 in high):
            features_count['Outer/High']  = features_count['Outer/High'] + 1
        if (words[i].lower() in low) or (words2 in low) or (words3 in low) or (words4 in low):
            features_count['Outer/Low']  = features_count['Outer/Low'] + 1
        if (words[i].lower() in objective) or (words2 in objective) or (words3 in objective) or (words4 in objective):
            features_count['Orientation/Objective']  = features_count['Orientation/Objective'] + 1
        if (words[i].lower() in subjective) or (words2 in subjective) or (words3 in subjective) or (words4 in subjective):
            features_count['Orientation/Subjective']  = features_count['Orientation/Subjective'] + 1
        if (words[i].lower() in implicit) or (words2 in implicit) or (words3 in implicit) or (words4 in implicit):
            features_count[ 'Manifestation/Implicit']  = features_count[ 'Manifestation/Implicit'] + 1
        if (words[i].lower() in explict) or (words2 in explict) or (words3 in explict) or (words4 in explict):
            features_count['Manifestation/Explicit']  = features_count['Manifestation/Explicit'] + 1
    
        features_count['Type/Modalisation'] = features_count['Modalisation/Probability'] + features_count['Modalisation/Usuality']
        features_count['Type/Modulation'] = features_count['Modulation/Readiness'] + features_count['Modulation/Obligation']
        features_count['Modality/Type'] = features_count['Type/Modalisation'] + features_count['Type/Modulation']
        
        features_count['Value/Outer'] = features_count['Outer/High'] + features_count['Outer/Low']
        features_count['Modality/Value'] = features_count['Value/Median'] + features_count['Value/Outer']
        
        features_count['Modality/Orientation'] = features_count['Orientation/Objective'] + features_count['Orientation/Subjective']
        
        features_count['Modality/Manifestation'] = features_count['Manifestation/Implicit'] + features_count['Manifestation/Explicit']
        
        values = []
        
    for k in feature_names:
        values.append(str(features_count[k]))
    new_file.write(",".join(values)+"\n")

new_file.close()
data_file.close()