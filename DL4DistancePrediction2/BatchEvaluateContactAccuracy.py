import cPickle
import sys
import os
import scipy.stats.mstats
import numpy as np

import config
import DistanceUtils
import ContactUtils
from MergePredictedContactMatrix import MergeAndSaveOneProtein

import getopt

def Usage():

    	print 'python BatchEvaluateContactAccuracy.py poteinList PKL_folder ground_truth_folder [fileSuffix]'
	print '  This script evaluate contact prediction accuracy for a list of proteins in their predicted dist or contact matrix files '
    	print '  PKL_folder: a folder containing predicted distance or contact matrix files with name like XXX.predictedDistMatrix.pkl or XXX.predictedContactMatrix.pkl'
    	print '     A predicted distance matrix file contains a tuple of 6 items: name, primary sequence, predicted distance prob matrix, predicted contact prob matrix, labelWeights, reference probabilities'
	print '  ground_truth_folder: folder for native distance matrix'
    	print '  file_suffix: suffix for the predicted dist/contact matrix file: .predictedDistMatrix.pkl (default) or .predictedContactMatrix.pkl'

def str_display(ls):
        if not isinstance(ls, (list, tuple, np.ndarray)):
                str_ls = '{0:.4f}'.format(ls)
                return str_ls

        str_ls = ['{0:.4f}'.format(v) for v in ls ]
        str_ls2 = ' '.join(str_ls)
        return str_ls2


def main(argv):


	if len(argv)<3:
		Usage()
		exit(1)

	proteinListFile = argv[0]
	predFolder = argv[1]
	nativefolder = argv[2]

	fileSuffix = ".predictedDistMatrix.pkl"

	if len(argv)>=4:
		fileSuffix = argv[3]

	if not os.path.isfile(proteinListFile):
		print 'the protein list file does not exist: ', proteinListFile
		exit(1)

	if not os.path.isdir(predFolder):
		print 'the folder for predicted matrix files does not exist: ', predFolder
		exit(1)

	if not os.path.isdir(nativefolder):
		print 'the folder for native distance matrix files does not exist: ', nativefolder
		exit(1)

	fh = open(proteinListFile, 'r')
	proteins = [ line.strip() for line in list(fh) ]
	fh.close()

	predictions = dict()
	for protein in proteins:
		predFile = os.path.join( predFolder, protein + fileSuffix ) 
		if not os.path.isfile(predFile):
			print 'the prediction file does not exist: ', predFile
			exit(1)

		fh = open(predFile, 'rb')
		pred = cPickle.load(fh)
		fh.close()
		
		if fileSuffix == '.predictedDistMatrix.pkl':
			predContactMatrix = pred[3]
		elif fileSuffix == '.predictedContactMatrix.pkl':
			predContactMatrix = pred['predContactMatrix']
		else:
			print 'unsupported file suffix for predicted files: ', fileSuffix
			exit(1)

		predictions[ protein ] = predContactMatrix

	if nativefolder is not None:
                print 'nativeFolder=', nativefolder
                avgacc, allacc = ContactUtils.EvaluateContactPredictions(predictions, nativefolder)
		print '******************average and detailed contact prediction accuracy*********************'
		for k, v in avgacc.iteritems():
			print 'average', k, str_display(v)
		for name, acc in allacc.iteritems():
			for k, v in acc.iteritems():
				print name, k, str_display(v)



if __name__ == "__main__":
    	main(sys.argv[1:])
