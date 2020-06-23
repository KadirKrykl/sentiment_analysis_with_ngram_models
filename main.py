import io, os
import re as re
import zipfile as zipfile
import sys
import random
import math 
import time
from functools import reduce

#Only for statistical report and data splitting
from sklearn.metrics import *

mytextzip = ''
docList={}
# {situation:{doc_id:{text:'text',tokens:[]}}}
idx_ID=1
situation = 0
docList[str(situation)] = {}
with zipfile.ZipFile('film_yorumlari.zip') as z:
    for zipinfo in z.infolist():
        mytextzip = ''
        if zipinfo.filename.endswith('.txt') and re.search('raw_texts', zipinfo.filename):
            with z.open(zipinfo) as f:
                textfile = io.TextIOWrapper(f, encoding='cp1254', newline='')
                for line in textfile:
                    if len(line.strip()): mytextzip += ' ' + line.strip()
                document={
                    'text' : mytextzip
                }
                docList[str(situation)][str(idx_ID)] = document
                if idx_ID % 35 == 0:
                    situation+=1
                    if situation != 3:
                        docList[str(situation)] = {}
                idx_ID+=1


def testDataSplit(wholeData,testCount):
    # testCount = test count for each class
    test_y = []
    test_x = []
    ids=[]
    for situation , docs in wholeData.items():
        docCount = len(docs)
        for i in range(testCount):
            randomID = str(random.randint(int(list(docs.keys())[0]),int(list(docs.keys())[-1])))
            while randomID in ids:
                randomID = str(random.randint(int(list(docs.keys())[0]),int(list(docs.keys())[-1])))
            ids.append(randomID)
            test_y.append(situation)
            test_x.append(docs[randomID]['text'])
    return test_x, test_y

#testX = [doc1_text,doc2_text,doc3_text,...]
#testY = [doc1_situation,doc2_situation,doc3_situation,...]
testX, testY = testDataSplit(docList,5)

# Non-breaking to normal space
NON_BREAKING = re.compile(u"\s+"), " "
# Multiple dot
MULTIPLE_DOT = re.compile(u"\.+"), "."
# Sentence dots
SENTENCE_DOT = re.compile(u"(?!\B\"\s+[^\"]*)[\.?!](?![^\"]*\s+\"\B)"), r" -d- "
# Merge multiple spaces.
ONE_SPACE = re.compile(r' {2,}'), ' '
# Numbers
NUMBERS= re.compile(r'[0-9]*[0-9]'), ' ' 
# 2.5 -> 2.5 - asd. -> asd . 
DOT_WITHOUT_FLOAT = re.compile("((?<![0-9])[\.])"), r' '
# 2,5 -> 2,5 - asd, -> asd , 
COMMA_WITHOUT_FLOAT = re.compile("((?<![0-9])[,])"), r' '
# doesn't -> doesn't  -  'Something' -> ' Something '
QUOTE_FOR_NOT_S = re.compile("\b(?<![n])[\'](?![t])\b"), r' '
AFTER_QUOTE_SINGLE_S = re.compile("\s+[s]\s+"), r' '
# Extra punctuations "!()
NORMALIZE = re.compile("([\–])"), r'-'
EXTRAS_PUNK = re.compile("([^\'\.\,\w\s\-\–])"), r' '

REGEXES = [
    NON_BREAKING,
    MULTIPLE_DOT,
    NUMBERS,
    QUOTE_FOR_NOT_S,
    AFTER_QUOTE_SINGLE_S,
    SENTENCE_DOT,
    DOT_WITHOUT_FLOAT,
    COMMA_WITHOUT_FLOAT,
    NORMALIZE,
    EXTRAS_PUNK,
    ONE_SPACE
]

def normalize(text):
    text = text.lower()
    for regexp, subsitution in REGEXES:
        text = regexp.sub(subsitution, text)
    return text

def tokenizer(text):
    normalizedText = normalize(text)
    tokens = normalizedText.split('-d-')
    tokens = "</s> <s>".join(tokens)
    tokens = tokens.split(' ')
    tokens[0] = "<s>"
    tokens[len(tokens)-1] = "</s>"
    return tokens

start_time = time.time()

for sit, docs in docList.items():
    docList[sit]['all_tokens'] = []
    for key,value in docs.items():
        if key != 'all_tokens':
            tokens = tokenizer(value['text'])
            docList[sit][key]['tokens'] = tokens
            docList[sit]['all_tokens'].extend(tokens)
    
elapsed_time = time.time() - start_time
print("Tokenize: "+str(elapsed_time))


# Getting 2 dictionary, like {abc:1},{abc:1} return {abc:2}
def reducer(first, last):
    for item in last: 
        for item in last:
            first[item] = first.get(item, 0) + last.get(item, 0)
    return first

# 1-GRAM MODEL
start_time = time.time()

for sit, docs in docList.items():
    a = docList[sit]['all_tokens']
    tokensMap = map(lambda char: dict([[char, 1]]), a)
    wordFreq = reduce(reducer, tokensMap)
    totalWord = sum(list(wordFreq.values())) #count all words
    docList[sit]['dl'] = totalWord 
    docList[sit]['TF'] = wordFreq  # words freq for all docs of each situation 
    docList[sit]['1-gram'] = dict()
    docList[sit]['1-gram'].update((k, v/totalWord) for k,v in docList[sit]['TF'].items())
    # 1-gram prob = TF(word) / totalWordCount
          
elapsed_time = time.time() - start_time
print("1-gram: "+str(elapsed_time))

start_time = time.time()

for sit, docs in docList.items():
    bifreq = map(lambda char: dict([[(docs['all_tokens'][char], docs['all_tokens'][char-1]),1]]), range(len(docs['all_tokens'])))
    bifreq = reduce(reducer, bifreq)
    docList[sit]['2TF'] = bifreq
    docList[sit]['2-gram'] = dict()
    docList[sit]['2-gram'].update((k, v/docList[sit]['TF'][k[1]]) for k,v in docList[sit]['2TF'].items())
    # prob for 2-gram = TF(word|word-1) / TF(word-1)

elapsed_time = time.time() - start_time
print("2-gram: "+str(elapsed_time))

start_time = time.time()

for sit, docs in docList.items():
    trigram = map(lambda char: dict([[(docs['all_tokens'][char], (docs['all_tokens'][char-1], docs['all_tokens'][char-2])),1]]), range(len(docs['all_tokens'])))
    trigram = reduce(reducer, trigram)
    trigram.update((k, v/docList[sit]['2TF'][k[1]]) for k,v in trigram.items())
    docList[sit]['3-gram'] = trigram
    # prob for 3-gram = TF(word|(word-1|word-2)) / TF(word-1|word-2)

elapsed_time = time.time() - start_time
print("3-gram: "+str(elapsed_time))

def classifier1gram(doc):
    tokens = tokenizer(doc) # doc splitting tokens
    probs = [0 for i in docList.keys()] # probs for situation
    for sit , properties in docList.items(): 
        for token in tokens:
            if token in properties['1-gram'].keys():
                probs[int(sit)] += math.log(1/properties['1-gram'][token])
    minVal = min(probs)
    return str(probs.index(minVal))
  

def classifier2gram(doc):
    tokens = tokenizer(doc)
    probs = [1 for i in docList.keys()]
    boundaries = list(map(lambda char: tuple([tokens[char], tokens[char-1]]), range(len(tokens)))) # boundaries created, like preprocess
    for sit , properties in docList.items():
        for boundary in boundaries:
            if boundary in properties['2-gram'].keys():
                probs[int(sit)] *= properties['2-gram'][boundary]
    maxVal = max(probs)
    return str(probs.index(maxVal))

def classifier3gram(doc):
    tokens = tokenizer(doc)
    probs = [1 for i in docList.keys()]
    boundaries = list(map(lambda char: tuple([tokens[char], (tokens[char-1],tokens[char-2])]), range(len(tokens)))) # boundaries created, like preprocess
    for sit , properties in docList.items():
        for boundary in boundaries:
            if boundary in properties['3-gram'].keys():
                probs[int(sit)] *= properties['3-gram'][boundary]
    maxVal = max(probs)
    return str(probs.index(maxVal))

predY_1gram=[]
for j in range(0,len(testX)):
    start_time = time.time()
    predY_1gram.append(classifier1gram(testX[j]))
    elapsed_time = time.time() - start_time
    print("Classifier: {0} \n Elapsed Time: {1} \n Class: {2} \n Predict: {3} \n".format('1-Gram',elapsed_time,testY[j],predY_1gram[-1]))

#Confusion Matrix 
conf_matrix_1gram=confusion_matrix(testY, predY_1gram)
classes = 'negative good neutral'.split()
print("\t",end='')
for label in classes:
    print("{:<10}".format(label),end='')
label=0
print()
for idx in conf_matrix_1gram:
    print("{:<10}".format(classes[label]),end='')
    for i in range(len(classes)):
        print("{:<10}".format(idx[i]),end='')
    label+=1
    print()

#Confusion Accruracy 
accurarcy=accuracy_score(testY, predY_1gram)
print("Accuracy for 1-gram model: " , str(accurarcy))

predY_2gram=[]
for j in range(0,len(testX)):
    start_time = time.time()
    predY_2gram.append(classifier2gram(testX[j]))
    elapsed_time = time.time() - start_time
    print("Classifier: {0} \n Elapsed Time: {1} \n Class: {2} \n Predict: {3} \n".format('2-Gram',elapsed_time,testY[j],predY_2gram[-1]))

#Confusion Matrix 
conf_matrix_2gram=confusion_matrix(testY, predY_2gram)
classes = 'negative good neutral'.split()
print("\t",end='')
for label in classes:
    print("{:<10}".format(label),end='')
label=0
print()
for idx in conf_matrix_2gram:
    print("{:<10}".format(classes[label]),end='')
    for i in range(len(classes)):
        print("{:<10}".format(idx[i]),end='')
    label+=1
    print()

#Confusion Accruracy 
accurarcy=accuracy_score(testY, predY_2gram)
print("Accuracy for 2-gram model: " , str(accurarcy))

predY_3gram=[]
for j in range(0,len(testX)):
    start_time = time.time()
    predY_3gram.append(classifier3gram(testX[j]))
    elapsed_time = time.time() - start_time
    print("Classifier: {0} \n Elapsed Time: {1} \n Class: {2} \n Predict: {3} \n".format('3-Gram',elapsed_time,testY[j],predY_3gram[-1]))

#Confusion Matrix 
conf_matrix_3gram=confusion_matrix(testY, predY_3gram)
classes = 'negative good neutral'.split()
print("\t",end='')
for label in classes:
    print("{:<10}".format(label),end='')
label=0
print()
for idx in conf_matrix_3gram:
    print("{:<10}".format(classes[label]),end='')
    for i in range(len(classes)):
        print("{:<10}".format(idx[i]),end='')
    label+=1
    print()

#Confusion Accruracy 
accurarcy=accuracy_score(testY, predY_3gram)
print("Accuracy for 3-gram model: " , str(accurarcy))