# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:30:07 2019

@author: Eduardo
"""
#import TokenizeDataSet as to
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize

train_n = 107
test_n = 109
dataset_path_train = "directory/speeches_"+str(train_n)+"_dwnominate_nonames_aux.txt"
dataset_path_test = "directory/speeches_"+str(test_n)+"_dwnominate_nonames_aux.txt"


new_train = "directory/speeches_"+str(train_n)+"_dwnominate_features_nonames.txt"
new_test = "directory/speeches_"+str(test_n)+"_dwnominate_features_nonames.txt"

def avgWordLength(dataset_path_train, dataset_path_test):
    
    features_train = []
    features_test = []
    tokenizer = RegexpTokenizer(r'\w+')
    data_file = open(dataset_path_train, 'r', encoding='latin-1')
    for line in data_file:
        
        line_split = line.split("|")        
        text = line_split[2]
        
        words = tokenizer.tokenize(text)
        
        nWords = len(words)
        sum_lengths = 0
        
        for w in words:
            sum_lengths += len(w)
        
        avgLength = sum_lengths/nWords
        features_train.append(avgLength)
    
    data_file = open(dataset_path_test, 'r', encoding='latin-1')
    for line in data_file:
        
        line_split = line.split("|")        
        text = line_split[2]
        
        words = tokenizer.tokenize(text)
        
        nWords = len(words)
        sum_lengths = 0
        
        for w in words:
            sum_lengths += len(w)
        
        avgLength = sum_lengths/nWords
        features_test.append(avgLength)
    
    return features_train, features_test

def avgNumberOfWords(dataset_path_train, dataset_path_test):
    
    features_train = []
    features_test = []
    
    data_file = open(dataset_path_train, 'r', encoding='latin-1')
    tokenizer = RegexpTokenizer(r'\w+')
   
    for line in data_file:
        
        line_split = line.split("|")        
        text = line_split[2]
        
        sentences = sent_tokenize(text)
        
        nSents = len(sentences)
        nWords = 0
        
        for s in sentences:
            
            words = tokenizer.tokenize(s)
            nWords += len(words)
        
        avgWords = nWords/nSents
        features_train.append(avgWords)
        
    data_file.close()
    
    data_file = open(dataset_path_test, 'r', encoding='latin-1')
    for line in data_file:
        
        line_split = line.split("|")        
        text = line_split[2]
        
        sentences = sent_tokenize(text)
        
        nSents = len(sentences)
        nWords = 0
        
        for s in sentences:
            
            words = tokenizer.tokenize(s)
            nWords += len(words)
        
        avgWords = nWords/nSents
        features_test.append(avgWords)
    
    return features_train, features_test
    
def charactersPerSentence(dataset_path_train, dataset_path_test):
    features_train = []
    features_test = []
    
    data_file = open(dataset_path_train, 'r', encoding='latin-1')
    tokenizer = RegexpTokenizer(r'\w+')
    for line in data_file:
        line_split = line.split("|")        
        text = line_split[2]
        
        sentences = sent_tokenize(text)
        
        nSents = len(sentences)
        nChars = 0
        
        for s in sentences:
            nChars += len(s)
        
        avgChars = nChars/nSents
        
        features_train.append(avgChars)
    data_file.close()
    
    data_file = open(dataset_path_test, 'r', encoding='latin-1')
    for line in data_file:
        line_split = line.split("|")        
        text = line_split[2]
        
        sentences = sent_tokenize(text)
        
        nSents = len(sentences)
        nChars = 0
        
        for s in sentences:
            nChars += len(s)
        
        avgChars = nChars/nSents
        
        features_test.append(avgChars)
    data_file.close()
    
    return features_train, features_test

def punctuationPerWord(dataset_path_train, dataset_path_test):
    features_train = []
    features_test = []
    
    data_file = open(dataset_path_train, 'r', encoding='latin-1')
    tokenizer = RegexpTokenizer(r'\w+')
    tokenizer_punct = RegexpTokenizer(r'[-,.:;\'"><]')
    for line in data_file:
        line_split = line.split("|")        
        text = line_split[2]
        
        words = tokenizer.tokenize(text)
        nWords = len(words)
        
        punct = tokenizer_punct.tokenize(text)
        npunct = len(punct)
        
        avgPunct = npunct/nWords
        
        features_train.append(avgPunct)
    data_file.close()
    
    data_file = open(dataset_path_test, 'r', encoding='latin-1')
    for line in data_file:
        line_split = line.split("|")        
        text = line_split[2]
        
        words = tokenizer.tokenize(text)
        nWords = len(words)
        
        punct = tokenizer_punct.tokenize(text)
        npunct = len(punct)
        
        avgPunct = npunct/nWords
        
        features_test.append(avgPunct)
    data_file.close()
    return features_train, features_test

def alphabeticCharacters(dataset_path_train, dataset_path_test):
    features_train = []
    features_test = []
    
    data_file = open(dataset_path_train, 'r', encoding='latin-1')
    tokenizer = RegexpTokenizer(r'\w+')
    
    for line in data_file:
        line_split = line.split("|")        
        text = line_split[2]
        
                   
        words = tokenizer.tokenize(text)  
        alphabetic = len(''.join(words))
        nChars = len(text)
        
        avgAlphabetic = alphabetic/nChars
        
        features_train.append(avgAlphabetic)
    data_file.close()
    
    data_file = open(dataset_path_test, 'r', encoding='latin-1')
    for line in data_file:
        line_split = line.split("|")        
        text = line_split[2]
        
                   
        words = tokenizer.tokenize(text)  
        alphabetic = len(''.join(words))
        nChars = len(text)
        
        avgAlphabetic = alphabetic/nChars
        
        features_test.append(avgAlphabetic)
    data_file.close()
    
    return features_train, features_test

def upperCase(dataset_path_train, dataset_path_test):
    features_train = []
    features_test = []
    
    data_file = open(dataset_path_train, 'r', encoding='latin-1')
    
    for line in data_file:
        line_split = line.split("|")        
        text = line_split[2]
        
        nUpper = 0
        for i in range(0,len(text)):
            if text[i].isupper():
                nUpper+=1
        
        avgUpper = nUpper/len(text)
        
        features_train.append(avgUpper)
    data_file.close()
    
    data_file = open(dataset_path_test, 'r', encoding='latin-1')
    for line in data_file:
        line_split = line.split("|")        
        text = line_split[2]
        
        nUpper = 0
        for i in range(0,len(text)):
            if text[i].isupper():
                nUpper+=1
        
        avgUpper = nUpper/len(text)
        
        features_test.append(avgUpper)
    data_file.close()
    return features_train, features_test

def digits(dataset_path_train, dataset_path_test):
    features_train = []
    features_test = []
    
    data_file = open(dataset_path_train, 'r', encoding='latin-1')
    
    for line in data_file:
        line_split = line.split("|")        
        text = line_split[2]
        
        ndigits = 0
        for i in range(0,len(text)):
            if text[i].isdigit():
                ndigits+=1
        
        avgdigit = ndigits/len(text)
        
        features_train.append(avgdigit)
    data_file.close()
    
    data_file = open(dataset_path_test, 'r', encoding='latin-1')
    for line in data_file:
        line_split = line.split("|")        
        text = line_split[2]
        
        ndigits = 0
        for i in range(0,len(text)):
            if text[i].isdigit():
                ndigits+=1
        
        avgdigit = ndigits/len(text)
        
        features_test.append(avgdigit)
    data_file.close()
    return features_train, features_test

def whiteSpaces(dataset_path_train, dataset_path_test):
    features_train = []
    features_test = []
    
    data_file = open(dataset_path_train, 'r', encoding='latin-1')
    
    for line in data_file:
        line_split = line.split("|")        
        text = line_split[2]
        
        nspaces = 0
        for i in range(0,len(text)):
            if text[i] == ' ':
                nspaces+=1
        
        avgspaces = nspaces/len(text)
        
        features_train.append(avgspaces)
    data_file.close()
    
    data_file = open(dataset_path_test, 'r', encoding='latin-1')
    for line in data_file:
        line_split = line.split("|")        
        text = line_split[2]
        
        nspaces = 0
        for i in range(0,len(text)):
            if text[i] == ' ':
                nspaces+=1
        
        avgspaces = nspaces/len(text)
        
        features_test.append(avgspaces)
    data_file.close()
    return features_train, features_test

def shortWords(dataset_path_train, dataset_path_test):
    
    features_train = []
    features_test = []
    tokenizer = RegexpTokenizer(r'\w+')
    data_file = open(dataset_path_train, 'r', encoding='latin-1')
    for line in data_file:
        
        line_split = line.split("|")        
        text = line_split[2]
        
        words = tokenizer.tokenize(text)
        
        nWords = len(words)
        nshort = 0
        for w in words:
            if len(w) < 4:
                nshort +=1
        features_train.append(nshort/nWords)
    data_file.close()
    
    data_file = open(dataset_path_test, 'r', encoding='latin-1')
    for line in data_file:
    
        line_split = line.split("|")        
        text = line_split[2]
        
        words = tokenizer.tokenize(text)
        
        nWords = len(words)
        nshort = 0
        for w in words:
            if len(w) < 4:
                nshort +=1
        features_test.append(nshort/nWords)
    data_file.close()
    
    return features_train, features_test

def functionWords(dataset_path_train, dataset_path_test):
    stop_words = stopwords.words('english')
    features_train = []
    features_test = []
    tokenizer = RegexpTokenizer(r'\w+')
    data_file = open(dataset_path_train, 'r', encoding='latin-1')
    
    for line in data_file:
        line_split = line.split("|")        
        text = line_split[2]
        
        words = tokenizer.tokenize(text)
        
        nWords = len(words)
        nFunct = 0
        for w in words:
            if w in stop_words:
                nFunct += 1
        features_train.append(nFunct/nWords)
    data_file.close()
    
    data_file = open(dataset_path_test, 'r', encoding='latin-1')
    
    for line in data_file:     
        line_split = line.split("|")        
        text = line_split[2]
        
        words = tokenizer.tokenize(text)
        
        nWords = len(words)
        nFunct = 0
        for w in words:
            if w in stop_words:
                nFunct += 1
        features_test.append(nFunct/nWords)
    data_file.close()
    return features_train, features_test

def wordRichness(dataset_path_train, dataset_path_test):
    stop_words = stopwords.words('english')
    features_train = []
    features_test = []
    tokenizer = RegexpTokenizer(r'\w+')
    data_file = open(dataset_path_train, 'r', encoding='latin-1')
    
    for line in data_file:
        line_split = line.split("|")        
        text = line_split[2]
        
        words = tokenizer.tokenize(text)
        
        nWords = len(words)
        vocab = []
        for w in words:
            if w.lower() not in vocab:
                vocab.append(w.lower())
        features_train.append(len(vocab)/nWords)
    data_file.close()
    
    data_file = open(dataset_path_test, 'r', encoding='latin-1')
    
    for line in data_file:     
        line_split = line.split("|")        
        text = line_split[2]
        
        words = tokenizer.tokenize(text)
        
        vocab = []
        for w in words:
            if w.lower() not in vocab:
                vocab.append(w.lower())
        features_test.append(len(vocab)/nWords)
    data_file.close()
    return features_train, features_test

tr = []
te = []
train_list = []
test_list = []
header = "dw1,dw2"

print("Calculating avg words length")
tr1, te1 = avgWordLength(dataset_path_train, dataset_path_test)
train_list.append(tr1)
test_list.append(te1)
header = header + ",avgWordLength"

print("Calculating nWords")
tr2, te2 = avgNumberOfWords(dataset_path_train, dataset_path_test)
train_list.append(tr2)
test_list.append(te2)
header = header + ",nWords"

print("Calculating nChars")
tr3, te3 = charactersPerSentence(dataset_path_train, dataset_path_test)
train_list.append(tr3)
test_list.append(te3)
header = header + ",nChars"

print("Calculating nPunct")
tr4, te4 = punctuationPerWord(dataset_path_train, dataset_path_test)
train_list.append(tr4)
test_list.append(te4)
header = header + ",nPunct"

print("Calculating alphabetic characters")
tr5, te5 = alphabeticCharacters(dataset_path_train, dataset_path_test)
train_list.append(tr5)
test_list.append(te5)
header = header + ",alphabeticChars"

print("Calculating uppercase characters")
tr6, te6 = upperCase(dataset_path_train, dataset_path_test)
train_list.append(tr6)
test_list.append(te6)
header = header + ",uppercaseChars"

print("Calculating digits")
tr7, te7 = digits(dataset_path_train, dataset_path_test)
train_list.append(tr7)
test_list.append(te7)
header = header + ",digits"

print("Calculating spaces")
tr8, te8 = whiteSpaces(dataset_path_train, dataset_path_test)
train_list.append(tr8)
test_list.append(te8)
header = header + ",nSpaces"

print("Calculating short words")
tr9, te9 = shortWords(dataset_path_train, dataset_path_test)
train_list.append(tr9)
test_list.append(te9)
header = header + ",shortWords"

print("Calculating function words")
tr10, te10 = functionWords(dataset_path_train, dataset_path_test)
train_list.append(tr10)
test_list.append(te10)
header = header + ",functionWords"

print("Calculating different words")
tr11, te11 = wordRichness(dataset_path_train, dataset_path_test)
train_list.append(tr11)
test_list.append(te11)
header = header + ",differentWords"

header = header+"\n"
for i in range(0,len(tr1)):
    #tr.append([tr1[i],tr2[i],tr3[i],tr4[i], tr5[i], tr6[i], tr7[i], tr8[i], tr9[i]])
    tr.append([x[i] for x in train_list])
for i in range(0,len(te1)):
    #te.append([te1[i],te2[i],te3[i],te4[i], te5[i], te6[i], te7[i], te8[i], tr9[i]])
    te.append([x[i] for x in test_list])
print("Generating files")
#header = "dw1,dw2,wordslength,nwords,nchars,nPunct,alphChar,upperCase,digits,whitespaces,shortWords\n"

train_file = open(new_train, 'w', encoding='latin-1')
test_file = open(new_test, 'w', encoding='latin-1')
data_file = open(dataset_path_train, 'r', encoding='latin-1')
train_file.write(header)
f = data_file.readlines()

for i in range(0,len(tr1)):
    line = f[i].split("|")
    train_file.write(line[0]+','+line[1])
    t = tr[i]
    for m in range(0,len(t)):
        train_file.write(','+str(t[m]))
    train_file.write("\n")
    
data_file.close()
train_file.close()

data_file = open(dataset_path_test, 'r', encoding='latin-1')
test_file.write(header)
f = data_file.readlines()

for i in range(0,len(te1)):
    line = f[i].split("|")
    test_file.write(line[0]+','+line[1])
    t = te[i]
    for m in range(0,len(t)):
        test_file.write(','+str(t[m]))
    test_file.write("\n")

data_file.close()    
test_file.close
