#!/usr/bin/env python

import sys
import numpy as np

usage = "python decisionStump.py <training csv file>  <test csv file>"

train_file = sys.argv[1]
test_file = sys.argv[2]

# v is for vector:
def countZerosAndOnes(v):
	classZero = 0
	classOne = 0
	for i in range(v):
		if i==0:
			classZero += 1
		elif i==1:
			classOne += 1
	p0 = classZero/n
	p1 = classOne/n
	return(p0,p1)
		
def chooseClassifier(x, y, Hy):
	maxInfoGain = 0
	bestFeature = 0 # How initialize?
	decisions = [] 
	for i in range(x):
		(uncertainty,pY0X0,pY1X0) = evaluateFeature(x, y, i)
		infoGain = Hy - uncertainty
		if infoGain > maxInfoGain:
			bestFeature = i
			maxInfoGain = infoGain
			print "i\tinfoGain"

		if pY0X0 > pY1X0:
			decisions[i] = 'y0_if_x0' 
		elif pY1X0 > pY0X0:
			decisions[i] = 'y1_if_x0'
		else: 
			decisions[i] = 'na'

	decision = decisions[bestFeature]

	return(bestFeature,decision)

def evaluateFeature(x, y, i):

	(px0,px1) = countZerosAndOnes(x[i])
	countY0X0 = 0
	countY1X0 = 0 
	countY0X1 = 0
	countY1X1 = 0
	countX0 = 0
	countX1 = 0
	for j in range(y):
		if x[i][j] == 0:
			countX0 += 1
			if y[j]==0:
				countY0X0 += 1
			if y[j]==1:
				countY1X0 += 1
		elif x[i][j] == 1:
			countX1 += 1
			if y[j]==0:
				countY0X1 += 1
			if y[j]==1:
				countY1X1 += 1

	pY0X0 = countY0X0/countX0
	pY1X0 = countY1X0/countX0 # = 1 - pY0X0, right?
	pY0X1 = countY0X1/countX1 
	pY1X1 = countY1X1/countX1 # = 1 - pY0X1

	Hyx0 = - pY0X0 * np.log2(pY0X0) - pY1X0 * np.log2(pY1X0)
	Hyx1 = - pY0X1 * np.log2(pY0X1) - pY1X1 * np.log2(pY1X1)

	uncertainty = px0 * Hyx0 + px1 * Hyx1
	return (uncertainty,pY0X0,pY1X0)

if __name__ == '__main__':

	data = np.genfromtxt(train_file, dtype=float, delimiter=',')

	y = data[:,1]  # Because y values are in the first column of the table
	cols = range(0,21) # Is this right?
	x = data[:,cols]   # What do here? # This takes all columns but the first one?
	xt = np.transpose(x)
	
	(n,m) = x.shape   # observations(rows), features(cols)                                                                                                                                                                   

# Train:
                                                                                                                                                                                         
	(py0,py1) = countZerosAndOnes(y)
	Hy = - py0 * np.log2(py0) - py1 * np.log2(py1)
	(bestFeature,decision) = chooseClassifier(x, y, Hy)

# Test:                                                                                                                                                                                                                                        
	test_data = np.genfromtxt(test_file, dtype=float, delimiter=',')
	y_test = test_data[:,1]
	x_test = test_data[:,bestFeature] #?                                                                                                                                                                                                   
	correct = 0
	total = len(y_test)
	
	pred_y = []

	for i in range(y):
		if decision == 'y0_if_x0':
			if x[i] == 0:
				pred_y[i] = 0
			elif x[i] == 1:
				pred_y[i] = 1
		elif decision == 'y1_if_x0':
			if x[i] == 0:
				pred_y[i] = 1
			elif x[i] == 1:
				pred_y[i] = 0
		
# Calculate accuracy:                                                                                                                                                                                                                  
	for i in range(y):
		if pred_y[i] == y[i]:
			correct += 1
			accuracy = correct/total
			print "Test Accuracy:",accuracy
				
