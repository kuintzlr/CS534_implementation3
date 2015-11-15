import decisionStump
import sys
import numpy as np
import random
from random import sample
import scipy
usage = "bagging.py\t<train.csv>\t<test.csv>\t<num stumps>"
num_samples = 40

def main():
    train_file, test_file, T = get_args()
    x_train, y_train = decisionStump.load_data(train_file)
    x_test, y_test = decisionStump.load_data(test_file)
    for t in range(5,T,5): 
        bags = [] #"(bestFeature, stump) ..."
        for i in range(t):
            bestFeature, stump = create_bag(x_train, y_train)
            bags.append((bestFeature, stump))
        test_acc = []
        train_acc = []
        for i in range(10000):        
            acc, c, i = compute_accuracy(x_train, y_train, bags)
            train_acc.append(acc)
            acc, c, i = compute_accuracy(x_test, y_test, bags)
            test_acc.append(acc)
        print str(t)+","+str(scipy.mean(train_acc))+","+str(scipy.mean(test_acc))

def compute_accuracy(x, y, bags):
    correct = 0.0
    for example, true_label in zip(x, y):
        if bag_predict(example, bags) == true_label:
            correct += 1
    return correct/len(x), correct, len(x) - correct
    

def bag_predict(example, bags):
    ones = 0
    for bag in bags:
        ones += decisionStump.predict(example, bag[1], bag[0])
    return round(float(ones)/len(bags))

def create_bag(x, y):
    sample_x, sample_y = zip(* sample(zip(x, y), num_samples))
    sample_x, sample_y = np.array(sample_x), np.array(sample_y)
    return decisionStump.train(sample_x, sample_y)

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
