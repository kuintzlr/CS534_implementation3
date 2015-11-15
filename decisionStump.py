#!/usr/bin/env python

import sys
import numpy as np
from math import log

# v is for vector:
usage = "decisionStump.py <training csv file>  <test csv file>"

def countZerosAndOnes(v, D):
    n = sum(D)    
    classOne, classZero = 0, 0
    for val, weight in zip(v, D):
        if val == 1:
            classOne += weight
        else:
            classZero += weight
    return classZero/float(n), classOne/float(n)
   
        
def chooseClassifier(x, y, Hy, D):
    maxInfoGain = 0
    bestFeature = 0 
    features = range(len(x[0]))
    decisions = [{0: -1, 1: -1} for i in features]
    for i in features:
        (uncertainty, pY0X0, pY1X0, pY0X1, pY1X1) = evaluateFeature(x, y, i, D)
        infoGain = Hy - uncertainty
        if infoGain > maxInfoGain:
            bestFeature = i
            maxInfoGain = infoGain

        if pY0X0 > pY1X0:
            decisions[i][0] = 0 
        else:
            decisions[i][0] = 1

        if pY0X1 > pY1X1 : 
            decisions[i][1] = 0
        else:
            decisions[i][1] = 1

    decision = decisions[bestFeature]
    return (bestFeature,decision)

def evaluateFeature(x, y, i, D):
    (px0, px1) = countZerosAndOnes(x.transpose()[i], D)
    
    countY0X0 = 0
    countY1X0 = 0 
    countY0X1 = 0
    countY1X1 = 0
    countX0 = 0
    countX1 = 0
    for j in range(len(y)):
        if x[j][i] == 0 and y[j] == 0:
            countX0 += 1 * D[j] 
            countY0X0 += 1 * D[j]
        elif x[j][i] == 1 and y[j] == 0:
            countX1 += 1 * D[j]
            countY0X1 += 1 *D[j]
        elif x[j][i] == 0 and y[j] == 1:
            countX0 += 1 * D[j]
            countY1X0 += 1 *D[j]
        else:
            countX1 += 1 * D[j]
            countY1X1 += 1 * D[j]
    
    try:
        pY0X0 = countY0X0/float(countX0)
    except ZeroDivisionError:
        pY0X0 = 1
    finally:
        pY1X0 = 1 - pY0X0
   
    try:
        pY0X1 = countY0X1/float(countX1)
    except ZeroDivisionError:
        pY0X1 = 1
    finally:
        pY1X1 = 1 - pY0X1
   
    try: 
        Hyx0 = -1 * pY0X0 * log(pY0X0, 2) - pY1X0 * log(pY1X0, 2)
    except ValueError:
        Hyx0 = 0    
    try: 
        Hyx1 = -1 * pY0X1 * log(pY0X1, 2) - pY1X1 * log(pY1X1, 2)
    except ValueError:
        Hyx1 = 0
    
    uncertainty = px0 * Hyx0 + px1 * Hyx1
    return (uncertainty,pY0X0,pY1X0, pY0X0, pY1X1)

def load_data(file):
    x, y = [], []
    for line in open(file, 'r'):
        line_arr = [int(val) for val in line.strip().split(",")]
        y.append(line_arr[0])
        x.append(line_arr[1:])
    return x, y


def main():
    train_file, test_file = get_cmd_args()
    x, y = load_data(train_file)
    x = np.array(x)
    y = np.array(y)
    bestFeature, stump = train(x, y)
    print "Best Feature:", bestFeature

    #test stuff        
    x_test, y_test = load_data(test_file)
    x_test, y_test = np.array(x_test), np.array(y_test)

    accuracy, t_positive, t_negative = compute_accuracy(x, y, stump, bestFeature)
    print accuracy, t_positive, t_negative
    accuracy, t_positive, t_negative = compute_accuracy(x_test, y_test, stump, bestFeature)
    print accuracy, t_positive, t_negative



def predict(x, stump, bestFeature):
    """
    Semantically nice.
    """
    return stump[x[bestFeature]]

#Train
def train(x, y, D=None):
    """
    
    Returns, index of best feature, decision stump
    """
    if D == None:
        D=[1] * len(x)
    py0, py1 = countZerosAndOnes(y, D) 
    if py0 == 0 or py1 == 0:
        Hy = 0
    else: 
        Hy = -1 * py0 * log(py0, 2) - py1 * log(py1, 2)
            
    (bestFeature, decision) = chooseClassifier(x, y, Hy, D)
    return bestFeature, decision

def compute_accuracy(x, y, stump, bestFeature):
    """
    x - features to evaluate
    y - true class label
    Hy - first uncertainty term
    decision - brute forced list of decisions to make based on features

    return accuracy, true positive, true negative
    """
    correct = 0
    total = len(y)
    pred_y = []
    for i, example in enumerate(x):
        decision = predict(example, stump, bestFeature)
        if decision == y[i]:
            correct += 1
    accuracy = correct / float(total)
    return accuracy, correct, float(total) - correct

def get_cmd_args():
    if len(sys.argv) < 3 or "-h" in sys.argv or "--help" in sys.argv:
        print >> sys.stderr, usage
        sys.exit()
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    return train_file, test_file

if __name__ == '__main__':
    main()
            
