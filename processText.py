"""
This library reads in all the Control and Dementia data from the Pitt Corpus
and puts it in a form that can be handled using NLP methods.

Tina Zhu and Emily Wu
April 2017
"""
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
import sys
from nltk.stem import *
from nltk.corpus import stopwords
from nltk import bigrams
import numpy as np

def removeSpaces(sentence):
    """
    This function removes spaces from the participant words that are in square
    brackets to prevent weird patterns from happening in the sentences when
    we split on spaces later on.
    """
    toReturn = "" # sentence being returned
    word = "" # word that needs to have the space removed
    inBracket = False # boolean to keep track of whether we're in a bracket

    for char in sentence:
        if char == "[":
            toReturn += char
            inBracket = True
        elif inBracket:
            if char != ' ':
                word += char
            if char == ']':
                toReturn += word
                word = ""
                inBracket = False
        else: # we are not in word
            toReturn += char

    return toReturn

def parse(category, keep, stem, stop=False, n=1):
    """
    This function gets all the files in a directory and reads the information
    into demographic and conversation dictionarys
    """
    mypath = "/scratch/ewu1/%s/" % category # pathway for the directory of
                                            # interest
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))] # getting
                                                                    # all files
                                                                    # in dir
    convo = {} # key: "participant-session"; value: list of lists containing
               # sentences
    demo = {} # key: "participant-session"; value: ["age", "gender"]

    for data in files: # looping through each file
        subjNum = data[0:3]
        sessionNum = data[4]
        with open(join(mypath, data)) as f:
            for i in range(5): # removing the first 5 lines of junk in header
                f.readline()

            info = f.readline().strip().split('|') # getting demographic info
            age = info[3].strip(';')
            gender = info[4]
            demo[subjNum+"-"+sessionNum] = [age, gender]

            # parse the actual text
            allWords = ""
            for sentence in f:
                sentence = removeSpaces(sentence)
                sentence = sentence.strip().split()
                if sentence[0] == "*PAR:": # only want to keep Ps lines
                    sentence = sentence[1:-1]
                    decode(sentence, keep, stem, stop, n)
                    allWords += " ".join(sentence) + " "
            convo[subjNum+"-"+sessionNum] = allWords
    return demo, convo

def decode(sentence, keep, stem, stop, n=1):
    """
    This function removes any unnecessary words from the conversation
    dictionary.
    @param: sentence - the sentence to be "decoded"
            keep - "alpha" for only English words (containing alphanumeric
            characters), "symbols" for only symbols from transcription process,
            "all" for both (i.e. all 'words')
            stem - True if want words to be stemmed
            stop - True if want to remove stopwords
            n - 1 if unigrams only, 2 if bigrams + unigrams, etc.
    """
    for i in range(len(sentence)):
        if sentence[i][0:2] == "[:" or sentence[i] == "[+exc]":
        # replace the word with [: with empty string
        # because the participant did not say it
        # or remove any instances of [+exc]
            sentence[i] = ''
        sentence[i] = sentence[i].strip("<")
        sentence[i] = sentence[i].strip(">")
        sentence[i] = sentence[i].strip(",")
        if n == 1:
            sentence[i] = sentence[i].strip(".")
        if re.search('[a-zA-Z]',sentence[i]):
            sentence[i] = sentence[i].replace("(","")
            sentence[i] = sentence[i].replace(")","")

    sentence[:] = [x for x in sentence if (x != '')]
    if keep == "alpha":
        sentence[:] = [x for x in sentence if re.search('[a-zA-Z]', x)]
    elif keep == "symbol":
        sentence[:] = [x for x in sentence if not re.search('[a-zA-Z]', x)]
    if stem == 1:
        reload(sys)
        sys.setdefaultencoding('utf-8')
        stemmer = SnowballStemmer("english")
        sentence[:] = [stemmer.stem(word) for word in sentence]

    sentence[:] = [x for x in sentence if not re.search(r'\\',x[0])]
    if stop:
        stops = stopwords.words('english')
        sentence[:] = [w for w in sentence if w.lower() not in stops]

def totalWordsinConvo(convo):
    """
    This function counts up the total words said by the participant in convo
    and converts it to a dictionary.
    """
    totalWords = {}

    for subj in convo:
        allWords = convo[subj].split(" ")
        totalWords[subj] = len(allWords)
    return totalWords

def avgWordsinConvo(convo):
    """
    This function computes the average length of all sentences said by a
    participant in convo and converts it to a dictionary.
    """
    avgWords = {}

    for subj in convo:
        allSent = convo[subj].split(".")
        totalSentences = len(allSent)
        totalWords = 0
        for sent in allSent:
            words = sent.split(" ")
            totalWords += len(words)
        average = float(totalWords)/totalSentences
        avgWords[subj] = average

    return avgWords

def getAges(demo):
    """
    Some files do not contain ages of Ps if the P's age has not changed from
    that of an earlier session. This function gets those ages and adds them to
    the demo dictionary, or replaces the age with None if that information is
    unavailable to us.
    """
    for line in demo:
        if demo[line][0] == '':
            try:
                demo[line][0] = demo[line[0:3]+'-0'][0]
            except KeyError:
                demo[line][0] = None

def debug(convo, demo):
    """
    This function allows us to debug code cleanly using print statements.
    """
    for subj in convo:
        print "\n***************************************"
        print "Subject Number: %s" % subj
        print "***************************************"
        print convo[subj]
    print

def makeUniqueDict(convo1, convo2, n=1):
    """
    This function creates a dictionary of unique words from convo.
    """
    allWords = []
    for convo in [convo1,convo2]:
        for subj in convo:
            if n == 1:
                words = convo[subj].split(" ")
                allWords += words
            elif n == 2:
                words = convo[subj].replace("?",".").split(".")
                # split by sentence
                for sentence in words:
                    unigrams = sentence.split(" ")
                    grams = list(bigrams(sentence.split(" ")))
                    for i in range(len(grams)):
                        grams[i] = grams[i][0] + " " + grams[i][1]
                    allWords += grams
                    allWords += unigrams

    allWords = set(allWords)

    uniqueWords = {}
    index = 0

    for word in allWords:
        if word != '':
            uniqueWords[word] = index
            index += 1

    return uniqueWords

def dictToList(convo):
    """
    This function converts a dictionary to a list.
    """
    temp = []

    for subj in convo:
        temp.append(convo[subj])

    return temp

def createFeatures(controlConvo, dementiaConvo, n=1):
    """
    This function creates the features for the data.
    n - 1 for unigrams only, 2 for bigrams + unigrams, etc.
    """
    uniqueDict = makeUniqueDict(controlConvo, dementiaConvo, n)
    features = dictToList(controlConvo) + dictToList(dementiaConvo)
    counts, vocab = createMatrix(uniqueDict, features)

    return counts.tocsr(), vocab

def createMatrix(uniqueDict, features):
    """
    This function creates the TfidfVectorizer for testing.
    """
    vec = TfidfVectorizer(vocabulary=uniqueDict)
    counts = vec.fit_transform(features)
    # count up all the different words in doc
    reversedVocab = dict((v,k) for k,v in vec.vocabulary_.iteritems())

    # freqs = [(word, counts.getcol(idx).sum()) for word, idx in vec.vocabulary_.items()]
    # lst = sorted(freqs, key = lambda x: -x[1])
    # for l in lst:
    #     print l

    return counts, reversedVocab

def main():
    """
    This function runs parseText.py.
    """
    # getting all Control P data
    controlDemo, controlConvo = parse("Control")

    getAges(controlDemo)

    debug(controlConvo, controlDemo)

    # getting all Dementia P data
    dementiaDemo, dementiaConvo = parse("Dementia")

    getAges(dementiaDemo)

    debug(dementiaConvo, dementiaDemo)

if __name__ == "__main__":
    main()
