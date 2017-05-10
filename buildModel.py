"""
This program takes in all the information from the Control and Dementia data
files and analyzes any relevant info.

Tina Zhu and Emily Wu
April 2017
"""
from sys import *
import heapq
from processText import *
from random import shuffle
from scipy.sparse import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cross_validation import train_test_split, KFold, \
            cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, \
                            make_scorer, f1_score, recall_score, precision_score
from sklearn.grid_search import GridSearchCV
import numpy as np
import scipy as sp

def usage():
    """
    prints out information for how program should be used on the command line.
    """
    print >> stderr, "Usage: python buildModel.py classifierType keep stem"

def checkArgv():
    """
    checks user input and checks to see if the input file exists
    """
    if len(argv) != 4:
        usage()
        exit()

def customScorer(trueLabels, predictLabels):
    print "F1 Score:", f1_score(trueLabels, predictLabels)
    print "Precision Score:", precision_score(trueLabels, predictLabels)
    print "Recall Score:", recall_score(trueLabels, predictLabels)
    print "Confusion Matrix:"
    print confusion_matrix(trueLabels, predictLabels)
    print

    return 1 # function was successful

def crossValidation(features, labels, classifierType, testSize=0.4):
    """
    This function performs cross validation.
    """
    if classifierType == "nb":
        model = MultinomialNB()
    elif classifierType == "dt":
        model = DecisionTreeClassifier()
    elif classifierType == "rf":
        model = RandomForestClassifier(n_estimators=300,criterion='gini',
        max_depth=30)
    elif classifierType == 'svm':
        model = LinearSVC()
    else:
        print "Options for classiferType: nb, dt, rf"

    sc = make_scorer(customScorer)
    scores = cross_val_score(model, features, labels, cv=10, scoring=sc)

def k(features,labels,model,modelType='df',modelInfo=False,vocab=None):
    """
    Runs 10-fold cross-validation.
    @param: features - matrix of features in non-sparse format
            labels - corresponding labels (dementia vs. control)
            model - initialized model to be tested
            modelType - type of model (algo. family)
            modelInfo - bool, print info on model? (eg. important features)
            vocab - list of vocab words (features)
    """
    labels = np.array(labels)
    kf = KFold(features.shape[0], n_folds=10,shuffle=True)
    f1s = []

    for train_index, test_index in kf:
        trainFeatures = features[train_index]
        testFeatures = features[test_index]

        trainLabels = labels[train_index]
        testLabels = labels[test_index]

        model.fit(trainFeatures, trainLabels)
        predictLabels = model.predict(testFeatures)
        f1 = f1_score(testLabels, predictLabels)
        print f1
        f1s.append(f1)
        print confusion_matrix(testLabels,predictLabels)
        if modelInfo and modelType == 'nb':
            printNBFeatures(model.feature_log_prob_,vocab,10)
        elif modelInfo and (modelType == 'rf' or modelType == "dt"):
            printRFDTFeatures(model.feature_importances_,vocab,10)
    return f1s

def tuneParam(features, labels, classifierType, vocab=None, display=None):
    """
    This function allows for tuning within a classifier.
    """
    features = features.tocsr()
    labels = np.array(labels)
    if classifierType not in ["nb", "dt", "rf", "svm"]:
        print "Options for classifierType: nb, dt, rf, svm"
        return

    leftOverFeatures, tuningFeatures, leftOverLabels, tuningLabels = \
        train_test_split(features, labels, test_size=0.2)

    bestModel = None
    bestF1Score = float("-inf")

    kf = KFold(leftOverFeatures.shape[0], n_folds=10,shuffle=True)

    for train_index, test_index in kf:
        trainFeatures = leftOverFeatures[train_index]
        testFeatures = leftOverFeatures[test_index]

        trainLabels = leftOverLabels[train_index]
        testLabels = leftOverLabels[test_index]

        if classifierType == "nb":
            model = MultinomialNB()
            parameters = {"alpha": [x / 100.0 for x in range(1, 101)]}
        elif classifierType == "dt":
            model = DecisionTreeClassifier()
            parameters = {"splitter": ["best", "random"], "max_depth": \
                          range(5, 20, 2)}
        elif classifierType == "rf":
            model = RandomForestClassifier()
            parameters = {"n_estimators": range(200, 401, 100), "max_depth": \
                          range(25, 51, 25), "n_jobs": [-1]}
        elif classifierType == 'svm':
            model = LinearSVC()

            parameters = {"C":[x / 100.0 for x in range(1, 200, 2)]}

        gridSearch = GridSearchCV(model, parameters, scoring="f1")
        gridSearch.fit(trainFeatures, trainLabels)
        predictLabels = gridSearch.predict(testFeatures)
        f1 = f1_score(testLabels, predictLabels)

        if f1 > bestF1Score:
            bestF1Score = f1
            bestModel = gridSearch

    predictLabels = bestModel.predict(tuningFeatures)

    customScorer(tuningLabels, predictLabels)


    if classifierType == "nb":
        print "Alpha: %.2f\n" % bestModel.best_estimator_.alpha

    elif classifierType == "dt":
        print "Splitter: %s" % bestModel.best_estimator_.splitter
        print "Max Depth: %d\n" % bestModel.best_estimator_.max_depth

    elif classifierType == "rf":
        print "Number of Estimators: %d" % bestModel.best_estimator_.n_estimators
        print "Max Depth: %d\n" % bestModel.best_estimator_.max_depth

    elif classifierType == "svm":
        print "C: %.2f\n" % bestModel.best_estimator_.C

    if display and vocab:
        if classifierType == "nb":
            printNBFeatures(bestModel.best_estimator_.feature_log_prob_, vocab, 10)
        elif classifierType == "rf" or classifierType == 'dt':
            printRFDTFeatures(bestModel.best_estimator_.feature_importances_, vocab, 10)
        elif classifierType == "svm":
            printSVMFeatures(bestModel.best_estimator_.coef_, vocab, 10)

    return f1_score(tuningLabels, predictLabels)

def printSVMFeatures(coef,vocab,n):
    """
    This function prints the n most important features (i.e. terms)
    for the SVM model to classify documents as either conversations
    with control participants or participants with dementia
    @param: coef - list of coefficients of the model
            vocab - list of vocabulary terms corresponding to features
            n - number of features to be printed
    """
    largeIndices = heapq.nlargest(n, range(len(coef[0])), coef[0].take)
    print "%d Most Likely Terms Given Class: Dementia" % n
    for i in largeIndices:
        print "%s: %.2f" % (vocab[i], coef[0][i])

    smallIndices = heapq.nsmallest(n, range(len(coef[0])), coef[0].take)
    print "\n%d Most LIkely Terms Given Class: Control" % n
    for i in smallIndices:
        print "%s: %.2f" % (vocab[i], coef[0][i])

def printRFDTFeatures(impt,vocab,n):
    """
    This function prints the n most important features (i.e. terms)
    for the RF model to classify documents as either conversations
    with control participants or participants with dementia
    @param: impt - list of feature importances
            vocab - list of vocabulary terms corresponding to features
            n - number of features to be printed
    """
    mostImpt = heapq.nlargest(10, \
                range(len(impt)), key=lambda x: impt[x])
    print "%d Most Important Terms" %(n)
    for index in mostImpt:
        print vocab[index], impt[index]

def printNBFeatures(probs,vocab,n):
    """
    This function prints the most informative terms for each class (dementia
    or control). A term is considered informative for the model if the ratio
    of its probability given class dementia over its probability given class
    control is very high.

    Likely terms have high probability i.e. lower magnitude log probability.
    @param: probs - log probabilities for each term given class control or
                    class dementia
            vocab - list of vocabulary terms corresponding to features
            n - number of features to be printed
    """
    ratios = [] # dementia over control probabilities
    control = probs[0]
    dementia = probs[1]
    for i in range(len(vocab)):
        ratios.append(10**(probs[1][i] - probs[0][i]))
    mostInformativeDementia = heapq.nlargest(n, \
                range(len(ratios)), key=lambda x: ratios[x])
    mostInformativeControl = heapq.nsmallest(n, \
                range(len(ratios)), key=lambda x: ratios[x])
    print "%d Most Informative Terms for Dementia" %(n)
    for index in mostInformativeDementia:
        print "%s, %0.2f" %(vocab[index], ratios[index])
    print "%d Most Informative Terms for Control" %(n)
    for index in mostInformativeControl:
        print "%s, %0.2f" %(vocab[index], ratios[index])

def tTest(dataset1, dataset2):
    """
    This function performs a t-test on the two datasets.
    """
    print sp.stats.ttest_ind(dataset1, dataset2, equal_var=False)

def getLists():
    return ALL_F1, ALL_PRES, ALL_REC

def baselineModels(dementiaDemo,dementiaConvo,controlConvo,controlDemo):
    totalWords = totalWordsinConvo(controlConvo).values() + \
                    totalWordsinConvo(dementiaConvo).values()
    avgWords = avgWordsinConvo(controlConvo).values() + \
                    avgWordsinConvo(dementiaConvo).values()

    labels = [0]*len(controlConvo) + [1]*len(dementiaConvo)
    features = np.column_stack((totalWords,avgWords))

    # model = DecisionTreeClassifier()
    model = LogisticRegression()
    results = k(features,labels,model)
    print " Mean F1 Score: %.2f" % np.mean(results)
    print " SD F1 Score: %.2f" % np.std(results)
    return results

def main():
    checkArgv()
    classifierType = argv[1]
    keep = argv[2]
    stem = int(argv[3])
    # getting all Control P data
    controlDemo, controlConvo = parse("Control",keep,stem)
    getAges(controlDemo)

    # getting all Dementia P data
    dementiaDemo, dementiaConvo = parse("Dementia",keep,stem)
    getAges(dementiaDemo)

    features, vocab = createFeatures(controlConvo, dementiaConvo)
    labels = [0]*len(controlConvo) + [1]*len(dementiaConvo)

    baselineModels(dementiaDemo,dementiaConvo,controlConvo,controlDemo)


if __name__ == "__main__":
    main()
