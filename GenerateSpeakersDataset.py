# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:33:58 2019

@author: Eduardo
"""

import os
from os import listdir
import csv
import matplotlib.pyplot as plt
import re
from nltk.corpus import words

vocab_path = 'dir/vocab.txt'

speech_number = '112'

path_speeches_map = "dir/"+speech_number+"_SpeakerMap.txt"
path_dwnominate = "dir/HS"+speech_number+"_members.csv"
path_speeches = "dir/speeches_"+speech_number+".txt"

path_speaker = "dir/speeches_"+speech_number+"_speakerId.txt"



def MapNominateSpeech(path_speeches_map, path_dwnominate):
    dwnominate_list = []
    speeches_map_list = []
    speeches_dict = {}
    speeches_dict2 = {}
    speaker_id = {}
    with open(path_dwnominate, 'r', encoding = 'utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader, None)
        
        for line_split in reader:
                                
            chamber = line_split[1]
            icpsr = line_split[2]
            bioname = line_split[9]
            bioname_split = bioname.split(',')
            nominate1 = line_split[13]
            nominate2 = line_split[14]
            
            if(chamber!='' and icpsr !='' and bioname !='' and nominate1!=''):
                dwnominate_list.append([chamber, icpsr, bioname, nominate1, nominate2])    
    
    with open(path_speeches_map, 'r', encoding = 'utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        next(reader, None)
        
        for line_split in reader:
            
            speakerid = line_split[0]
            speechid = line_split[1]
            last_name = line_split[2]
            name = line_split[3]
            chamber = line_split[4]
            
            if(speakerid !='' and speechid!='' and last_name !='' and name !='' and chamber !=''):
              speeches_map_list.append([speakerid, speechid, last_name, name, chamber])
    
    n = 0
    t = len(dwnominate_list)
    for i in dwnominate_list:
        
        bioname_split = i[2].split(',')
        last_name = bioname_split[0]
        bioname_split.pop(0)
        firstname = " ".join(bioname_split)
        nominate1 = i[3]
        nominate2 = i[4]
        
        matched = 0
        last_name2 = replaceTilde(last_name.lower())
        firstname2 = replaceTilde(firstname.lower())
        for j in speeches_map_list:               
            new_lastname = replaceTilde(j[2].lower())
            new_name = replaceTilde(j[3].lower())
            
            if new_lastname == last_name2:
                if(new_name == firstname2):
                    speeches_dict[j[1]] = nominate1
                    speaker_id[j[i]] = j[0]
                    speeches_dict2[j[1]] = nominate2
                    matched = 1
                else:
                    if new_name in firstname2:
                        speeches_dict[j[1]] = nominate1
                        speaker_id[j[1]] = j[0]
                        speeches_dict2[j[1]] = nominate2
                        matched = 1
                    else:
                        if new_name[0] == firstname2[1] and (j[1] not in speeches_dict.keys()):
                            speeches_dict[j[1]] = nominate1
                            speaker_id[j[1]] = j[0]
                            speeches_dict2[j[1]] = nominate2
                            matched = 1
            else:
                
                if ((new_lastname in last_name2) or last_name2 in new_lastname):         
                    if(new_name == firstname2):
                        speeches_dict[j[1]] = nominate1
                        speaker_id[j[1]] = j[0]
                        speeches_dict2[j[1]] = nominate2
                        matched = 1
                    else:
                        if new_name in firstname2:
                            speeches_dict[j[1]] = nominate1
                            speaker_id[j[1]] = j[0]
                            speeches_dict2[j[1]] = nominate2
                            matched = 1
                        else:
                            if new_name[0] == firstname2[1] and (j[1] not in speeches_dict.keys()):
                                speeches_dict[j[1]] = nominate1
                                speaker_id[j[1]] = j[0]
                                speeches_dict2[j[1]] = nominate2
                                matched = 1
        if(matched == 0):
            print("no match in line " + str(n))
            print(firstname.lower())
        n += 1
        print(str(n) + '/' + str(t))
    csvfile.close()
    return speeches_dict, speeches_dict2, speaker_id

def GenerateSpeakers(speeches_dict,speeches_dict2, speaker_id, path_speeches, path_speaker, useDictionary = False):
    speeches_file = open(path_speeches, 'r', encoding='latin-1')
    new_file = open(path_speaker, 'w')
    speeches_file.readline()
    i = 0
    for line in speeches_file:
        i+=1
        print(i)
        line_split = line.split("|")
        speech_id = line_split[0]
        text = line_split[1]
        vocab = []
        text_clean = []
        text_split = text.split(" ")    
        if speech_id in speeches_dict.keys():
            
            for word in text_split:
                if (word.lower().replace(".",'') not in vocab):
                    if word.islower():
                        vocab.append(word.lower().replace(".",''))
            for word in text_split:
                if word.lower().replace(".",'') in vocab:
                    text_clean.append(word.replace(".",''))
            #if len(text_clean)>0:
            if len(" ".join(text_clean)) > 1000:
                
                new_line_list = [speech_id, speaker_id[speech_id]]
                new_line = "|".join(new_line_list)
                if ("\n" not in new_line):
                    new_line = new_line + "\n"
                new_file.write(new_line)
                if "\n" not in text:
                    text = text+"\n"               
    new_file.close()
    speeches_file.close()

def generateDataset(path_speeches_map, path_dwnominate,  path_speeches, path_speaker):
    d1, d2, sd =  MapNominateSpeech(path_speeches_map, path_dwnominate)
    GenerateSpeakers(d1,d2, sd, path_speeches, path_speaker, useDictionary = False)

def replaceTilde(name):
    vowels = ['a','e','i','o','u']
    vowels2 = ['á','é','í','ó','ú']
    
    for i in range(0, len(vowels)):
        name = name.replace(vowels2[i], vowels[i])
        
    return name