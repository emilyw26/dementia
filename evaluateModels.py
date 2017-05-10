"""
This program runs buildModel.py with different parameters (for both the model
and the features) in order to evaluate which models are most effective on
a validation set. It also runs statistical tests on the results to see if
differences are statistically significant.

Tina Zhu and Emily Wu
April 2017
"""

from sys import *
from processText import *
from buildModel import *
from test import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, \
                        make_scorer, f1_score, recall_score, precision_score
import numpy as np
import scipy as sp

def tTest(dataset1, dataset2):
    """
    This function performs a t-test on the two datasets.
    """
    tValue, pValue = sp.stats.ttest_ind(dataset1, dataset2, equal_var=False)
    print "\nt-value: %5.2f" % (tValue)
    print "p-value: %5.3f" % (pValue)

    if pValue <= 0.05:
        print "\nThe result is significant.\n"
    else:
        print "\nThe result is not significant.\n"

def baseline():
    """
    This function gets the baseline metrics.
    """
    t = Test("nb", "all", 0)
    controlDemo, controlConvo = parse("Control", t.getKeep(), t.getStem())

    # getting all Dementia P data
    dementiaDemo, dementiaConvo = parse("Dementia", t.getKeep(), t.getStem())
    print "\nBaseline test of average and total words in a document.\n"
    f1s = baselineModels(dementiaDemo,dementiaConvo,controlConvo,controlDemo)
    print "Average F1 score: %.2f" % (np.mean(f1s))
    print "SD for F1 scores: %.2f" % (np.std(f1s))

def makeData(t):
    """
    This function makes the data needed for the tests based on the
    attributes of the test t.
    """
    controlDemo, controlConvo = parse("Control", t.getKeep(), t.getStem(), \
                                        t.getStop(), t.getNGrams())

    dementiaDemo, dementiaConvo = parse("Dementia", t.getKeep(), t.getStem(), \
                                        t.getStop(), t.getNGrams())

    features, vocab = createFeatures(controlConvo, dementiaConvo, t.getNGrams())
    labels = [0]*len(controlConvo)+[1]*len(dementiaConvo)

    return controlDemo, controlConvo, dementiaDemo, dementiaConvo, features, \
           vocab, labels

def printScores(f1List1, f1List2, name1, name2):
    """
    This function prints out statistical info for the two f1 lists.
    """
    print "%s F1 Score Avg: %.2f" % (name1, np.mean(f1List1))
    print "%s F1 Score SD: %.2f" % (name1, np.std(f1List1))
    print "%s F1 Score: %.2f" % (name2, np.mean(f1List2))
    print "%s F1 Score SD: %.2f" % (name2, np.std(f1List2))

def noAlpha(modelName, modelStr, t1, t2, l):
    """
    This function compares symbol data to alpha data.
    """
    symbolFeatures = l[0]
    symbolLabels = l[1]
    symbolVocab = l[2]
    alphaFeatures = l[3]
    alphaLabels = l[4]
    alphaVocab = l[5]

    t1.setModel(modelName)
    t2.setModel(modelName)

    print "\nRunning %s on non-alphanumeric, un-stemmed data.\n" % modelStr
    symbolF1s = k(symbolFeatures, symbolLabels, t1.getModel(), \
                t1.getClassifierType(), t1.getDisplay(),symbolVocab)

    print "\nRunning %s on alphanumeric, un-stemmed data.\n" % modelStr
    alphaF1s = k(alphaFeatures, alphaLabels, t2.getModel(), \
                t2.getClassifierType(), t2.getDisplay(), alphaVocab)

    printScores(symbolF1s, alphaF1s, "Symbol", "Alpha")
    tTest(symbolF1s, alphaF1s)

def noAlphaTest():
    """
    This function runs noAlpha for each model.
    """
    t1 = Test(None, "symbol", 0, False)

    symbolControlDemo, symbolControlConvo, symbolDementiaDemo, \
    symbolDementiaConvo, symbolFeatures, symbolVocab, symbolLabels = makeData(t1)

    t2 = Test(None, "alpha", 0, True)

    alphaControlDemo, alphaControlConvo, alphaDementiaDemo, alphaDementiaConvo,\
    alphaFeatures, alphaVocab, alphaLabels = makeData(t2)

    l = []
    l.append(symbolFeatures)
    l.append(symbolLabels)
    l.append(symbolVocab)
    l.append(alphaFeatures)
    l.append(alphaLabels)
    l.append(alphaVocab)

    noAlpha("nb", "Naive Bayes", t1, t2, l)
    noAlpha("dt", "Decision Trees", t1, t2, l)
    noAlpha("rf", "Random Forest", t1, t2, l)
    noAlpha("svm", "Support Vector Machines", t1, t2, l)

def runTest(model1, model2, t1, t2, l, comp, runTTest=True):
    """
    This function compares one dataset to another one using tuning.
    param: comp - names for the two things being compared (e.g.
           "Stemmed" vs. "Unstemmed")
    """
    t1.setModel(model1)
    t2.setModel(model2)

    features1 = l[0]
    labels1 = l[1]
    vocab1 = l[2]
    features2 = l[3]
    labels2 = l[4]
    vocab2 = l[5]

    f1s1 = []
    f1s2 = []

    print "\nTesting %s Data.\n" % comp[0]

    for i in range(10):
        f1s1.append(tuneParam(features1, labels1, model1, vocab1, t1.getDisplay()))

    print "\nTesting %s Data.\n" % comp[1]

    for i in range(10):

        f1s2.append(tuneParam(features2, labels2, model2, vocab2, t2.getDisplay()))

    printScores(f1s1, f1s2, comp[0], comp[1])

    if runTTest:
        tTest(f1s1, f1s2)

    return f1s1, f1s2

def stemmingTest():
    """
    This function tests the stemmed vs unstemmed data comparison for all 4
    models.
    """
    t1 = Test(None, "alpha", 0)

    usControlDemo, usControlConvo, usDementiaDemo, usDementiaConvo,\
    usFeatures, usVocab, usLabels = makeData(t1)

    t2 = Test(None, "alpha", 1)

    sControlDemo, sControlConvo, sDementiaDemo, sDementiaConvo,\
    sFeatures, sVocab, sLabels = makeData(t2)

    l = []
    l.append(usFeatures)
    l.append(usLabels)
    l.append(usVocab)
    l.append(sFeatures)
    l.append(sLabels)
    l.append(sVocab)

    runTest("nb", "nb", t1, t2, l,("Unstemmed","Stemmed"))

def stopWordsTest():
    """
    This function tests the data w/stopwords & w/o comparison for all 4 models.
    """
    t1 = Test(None, "alpha", 1, stop=True)

    usControlDemo, usControlConvo, usDementiaDemo, usDementiaConvo,\
    usFeatures, usVocab, usLabels = makeData(t1)

    t2 = Test(None, "alpha", 1, stop=False)

    sControlDemo, sControlConvo, sDementiaDemo, sDementiaConvo,\
    sFeatures, sVocab, sLabels = makeData(t2)

    l = []
    l.append(usFeatures)
    l.append(usLabels)
    l.append(usVocab)
    l.append(sFeatures)
    l.append(sLabels)
    l.append(sVocab)

    print "Number of Unique Words in Vocabulary"
    print "-- excluding stopwords:", len(usVocab)
    print "-- including stopwords:", len(sVocab)

    runTest("nb", "nb", t1, t2, l,("Excl. Stopwords","Incl. Stopwords"))
    runTest("rf", "rf", t1, t2, l,("Excl. Stopwords","Incl. Stopwords"))
    runTest("dt", "dt", t1, t2, l,("Excl. Stopwords","Incl. Stopwords"))
    runTest("svm", "svm", t1, t2, l,("Excl. Stopwords","Incl. Stopwords"))

def nGramsTest():
    """
    This function tests whether including bigrams and trigrams improves
    results.
    """
    t1 = Test(None, "alpha", 1, nGrams=1)

    usControlDemo, usControlConvo, usDementiaDemo, usDementiaConvo,\
    usFeatures, usVocab, usLabels = makeData(t1)

    t2 = Test(None, "alpha", 1, nGrams=2)

    sControlDemo, sControlConvo, sDementiaDemo, sDementiaConvo,\
    sFeatures, sVocab, sLabels = makeData(t2)

    l = []
    l.append(usFeatures)
    l.append(usLabels)
    l.append(usVocab)
    l.append(sFeatures)
    l.append(sLabels)
    l.append(sVocab)

    print "Number of Unique Words in Vocabulary"
    print "-- with unigrams:", len(usVocab)
    print "-- with unigrams + bigrams:", len(sVocab)

    runTest("nb", "nb", t1, t2, l,("Unigrams","Unigrams + Bigrams"))
    runTest("dt", "dt", t1, t2, l,("Unigrams","Unigrams + Bigrams"))
    runTest("rf", "rf",  t1, t2, l,("Unigrams","Unigrams + Bigrams"))
    runTest("svm", "svm", t1, t2, l,("Unigrams","Unigrams + Bigrams"))

def compareModels():
    """
    This function compares tuned model classification for stemmed, alpha
    and symbol data.
    """
    t1 = Test(None, "alpha", 1, nGrams = 1)
    t2 = Test(None, "alpha", 1, nGrams = 1)

    controlDemo, controlConvo, dementiaDemo, dementiaConvo, \
    features, vocab, labels = makeData(t1)

    l = []
    l.append(features)
    l.append(labels)
    l.append(vocab)
    l.append(features)
    l.append(labels)
    l.append(vocab)

    print "\nGetting tuned F1 scores for all models."

    nbF1s, dtF1s = runTest("nb", "dt", t1, t2, l, ("Naive Bayes", "Decision Trees"), False)
    rfF1s, svmF1s = runTest("rf", "svm", t1, t2, l, ("Random Forest", "Support Vector Machines"), False)

    print "t-Test for Naive Bayes and Decision Trees:"
    tTest(nbF1s, dtF1s)

    print "t-Test for Naive Bayes and Random Forest:"
    tTest(nbF1s, rfF1s)

    print "t-Test for Naive Bayes and SVMs:"
    tTest(nbF1s, svmF1s)

    print "t-Test for Decision Trees and Random Forest:"
    tTest(dtF1s, rfF1s)

    print "t-Test for Decision Trees and SVMs:"
    tTest(dtF1s, svmF1s)

    print "t-Test for Random Forest and SVMs:"
    tTest(rfF1s, svmF1s)

def seeFeatures():
    """
    This function allows the user to print out the useful features for each model.
    """
    t = Test(None, "alpha", 1, True)

    controlDemo, controlConvo, dementiaDemo, dementiaConvo, features, vocab, labels = makeData(t)

    """
    # Naive Bayes
    print "\nNaive Bayes features:"
    tuneParam(features, labels, "nb", vocab, True)

    # Decision Trees
    print "\nDecision Tree features:"
    tuneParam(features, labels, "dt", vocab, True)

    # Random Forest
    print "\nRandom Forest features:"
    tuneParam(features, labels, "rf", vocab, True)
    """

    # SVMs
    print "\nSVM features:"
    tuneParam(features, labels, "svm", vocab, True)

def main():
    test = raw_input("Which test do you want to run? ")
    if test == "baseline":
        baseline()
    elif test == "noAlpha":
        noAlphaTest()
    elif test == "stemming":
        stemmingTest()
    elif test == "stopwords":
        stopWordsTest()
    elif test == "ngrams":
        nGramsTest()
    elif test == "compare":
        compareModels()
    elif test == "features":
        seeFeatures()
    else:
        # test to be run
        seeFeatures()


main()
