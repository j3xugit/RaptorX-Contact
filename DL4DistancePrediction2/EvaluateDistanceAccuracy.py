import cPickle
import sys
import os
import scipy.stats.mstats
import numpy as np

import config
import DistanceUtils
#import ContactUtils
#from MergePredictedContactMatrix import MergeAndSaveOneProtein

import getopt

def Usage():

    	print 'python EvaluateDistanceAccuracy.py bound_PKL ground_truth_PKL'
	print '  This script evaluate distance bound accuracy for a protein in its predicted bound matrix file '
    	print '  bound_PKL: a predicted distance bound file with name like XXX.bound.pkl'
    	print '     A predicted distance bound matrix file contains a tuple of 3 items: bound, name, primary sequence'
	print '     bound is a dict() and each item is a matrix with dimension L*L*10 where L is sequence length, the 1st entry is the estimated inter-atom distance and the remaining are a variety of deviations'
	print '  ground_truth_PKL: a native distance matrix file with name like XXX.atomDistMatrix.pkl'
	print '  This script will output absolute error of distance prediction, relative error, precision, recall, F1 and GDT'



def main(argv):


	if len(argv)<2:
		Usage()
		exit(1)

	predFile = argv[0]
	nativeFile = argv[1]

	if not os.path.isfile(predFile):
		print 'the predicted bound matrix file does not exist: ', predFile
		exit(1)

	if not os.path.isfile(nativeFile):
		print 'the native distance matrix file does not exist: ', nativeFile
		exit(1)

	fh = open(predFile, 'rb')
	pred = cPickle.load(fh)
	fh.close()


	fh = open(nativeFile, 'rb')
	native = cPickle.load(fh)
	fh.close()		

	newPred = pred[0]
	newPred['seq'] = pred[2]
        acc = DistanceUtils.EvaluateDistanceBoundAccuracy(pred[0], native)

	target = os.path.basename(nativeFile)
	for apt, value in acc.iteritems():
		print target, apt, value



if __name__ == "__main__":
    	main(sys.argv[1:])
