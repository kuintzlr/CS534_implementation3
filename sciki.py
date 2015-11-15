#!/usr/bin/env python
from sklearn import tree
import decisionStump
import sys
from sklearn import ensemble
from random import seed
seed(1010)

#sklearn.tree.DecisionTreeClassifier


def compute_accuracy(x, y, classifier):
    """
    x - features to evaluate
    y - true class label
    Hy - first uncertainty term
    decision - brute forced list of decisions to make based on features

    return accuracy, true positive, true negative
    """
    correct = 0
    total = len(y)
    for i, example in enumerate(x):
        decision = classifier.predict(example)
        if decision == y[i]:
            correct += 1
    accuracy = correct / float(total)
    return accuracy, correct, float(total) - correct


data = [[1], [2]]
y = [1,0]


X_train, y_train = decisionStump.load_data(sys.argv[1])
X_test, y_test = decisionStump.load_data(sys.argv[2])
classifier = tree.DecisionTreeClassifier(max_depth=1)
stump = tree.DecisionTreeClassifier(max_depth=1)

trained = classifier.fit(X_train, y_train)
print "Decision stump"
print compute_accuracy(X_train, y_train, classifier)
print compute_accuracy(X_test, y_test, classifier)

print "Bagged results"
bags = ensemble.BaggingClassifier(base_estimator=stump, n_estimators=10, max_samples=40)
bag_trained = bags.fit(X_train, y_train)
print compute_accuracy(X_train, y_train, bag_trained)
print compute_accuracy(X_test, y_test, bag_trained)

print "Adaboost results"
boost = ensemble.AdaBoostClassifier(base_estimator=stump, n_estimators=10)
ada = boost.fit(X_train, y_train)
print compute_accuracy(X_train, y_train, ada)
print compute_accuracy(X_test, y_test, ada)


