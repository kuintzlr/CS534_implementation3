#!/usr/bin/env python
import decisionStump
import sys
import numpy as np
from math import log, exp

usage = "adaboost.py\t<train.csv>\t<test.csv>\t<ensemble size>"

def main():
    train, test, L = get_args()
    X_train, y_train = decisionStump.load_data(train)
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    X_test, y_test = decisionStump.load_data(test)
    X_test, y_test = np.array(X_test), np.array(y_test)
    ensemble = ada_boosting(X_train, y_train, L)
    print compute_accuracy(X_train, y_train, ensemble)
#    print compute_accuracy(X_test, y_test, ensemble)


def predict(X, ensemble):
    total = 0
    for e in ensemble:
        if decisionStump.predict(X, e["stump"], e["bestFeature"]) == 0:
            total += e["alpha"] * -1
        else:
            total += e["alpha"]  
    return 1 if total > 0 else 0

def compute_accuracy(X, y, ensemble):
    correct = 0
    for example, true_label in zip(X, y):
        if predict(example, ensemble) == true_label:
            correct += 1
    return correct/float(len(X)), correct, len(X) - correct

def ada_boosting(X, y, L):
    D = [1.0/len(X)] * len(X)
    
    ensemble = [] #{stump": => , "alpha" : => } 
    for i in range(L):
        errorSum = 0
        bestFeature, stump = decisionStump.train(X, y, D)
        print bestFeature
        #calculate error sum
        for i, (example, true_label) in enumerate(zip(X, y)):
            if decisionStump.predict(example, stump, bestFeature) != true_label:
                errorSum += D[i]
        mean_error = errorSum / sum(D)
        if mean_error == 0:
            print >> sys.stderr, "achieved perfect training, exiting at iteration:", L
            break
        alpha = .5 * log((1 - mean_error) / mean_error)
        #update weights
        for i, (example, true_label) in enumerate(zip(X, y)):
            if decisionStump.predict(example, stump, bestFeature) == true_label:
                print "corect", exp(-1 * alpha)
                D[i] *= exp(-1 * alpha)
            else:
                print "incorrect", exp(alpha)
                D[i] *= exp(alpha)
        print i, mean_error, alpha
        ensemble.append({"stump": stump, "bestFeature": bestFeature, "alpha": alpha})
    return ensemble
                
def get_args():
    if len(sys.argv) < 4 or "-h" in sys.argv or "--help" in sys.argv:
        print >> sys.stderr, usage
        sys.exit()
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    T = int(sys.argv[3])
    return train_file, test_file, T


if __name__ == "__main__":
    main()
