"""
This object creates a class that contains all the information about what
analyses to run for our project.

Emily Wu and Tina Zhu
April 2017
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz

class Test(object):
    """
    A class containing information
    """
    def __init__(self, classifierType, keep, stem, display=False, \
                            stop=False, nGrams=1):
        self.classifierType = classifierType
        self.keep = keep
        self.stem = stem
        self.display = display 
        self.stop = stop
        self.nGrams = nGrams
        if self.classifierType == "nb":
            self.model = MultinomialNB()
        elif self.classifierType == "dt":
            self.model = DecisionTreeClassifier()
        elif self.classifierType == "rf":
            self.model = RandomForestClassifier()
        elif self.classifierType == 'svm':
            self.model = LinearSVC()
        else:
            self.model = None

    def getClassifierType(self):
        return self.classifierType

    def getStop(self):
        return self.stop

    def getNGrams(self):
        return self.nGrams

    def getKeep(self):
        return self.keep

    def getStem(self):
        return self.stem

    def getModel(self):
        return self.model

    def getDisplay(self):
        return self.display

    def setClassifierType(self, newClassifierType):
        self.classifierType = newClassifierType

    def setKeep(self, newKeep):
        self.keep = newKeep

    def setStem(self, newStem):
        self.stem = newStem

    def setModel(self, name):
        self.classifierType = name
        if self.classifierType == "nb":
            self.model = MultinomialNB()
        elif self.classifierType == "dt":
            self.model = DecisionTreeClassifier()
        elif self.classifierType == "rf":
            self.model = RandomForestClassifier()
        elif self.classifierType == 'svm':
            self.model = LinearSVC()
        else:
            self.model = None

    def setDiplay(self, display):
        self.display = display