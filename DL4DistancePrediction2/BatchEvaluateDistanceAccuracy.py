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

    	print 'python BatchEvaluateDistanceAccuracy.py poteinList bound_PKL_folder ground_truth_folder [minSeqSep]'
	print '  This script evaluate distance bound accuracy for a list of proteins in their predicted bound matrix files '
    	print '  bound_PKL_folder: a folder containing predicted distance bound with name like XXX.bound.pkl'
    	print '     A predicted distance bound matrix file contains a tuple of 3 items: bound, name, primary sequence'
	print '     bound is a dict() and each item is a matrix with dimension L*L*10 where L is sequence length, the 1st entry is the estimated inter-atom distance and the remaining are a variety of deviations'
	print '	 minSeqSep: optional. The minimum sequence separation between two residues for which its distance is evaluated. default 12'
	print '  This script will output absolute error of distance prediction, relative error, precision, recall, F1 and GDT'


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
	fileSuffix = '.bound.pkl'

	minSeqSep = 12
	if len(argv)>=4:
		minSeqSep = np.int32(argv[3])
	if minSeqSep < 2:
		print 'the minimum sequence separation shall be at least 2'
		exit(1)

	if not os.path.isfile(proteinListFile):
		print 'the protein list file does not exist: ', proteinListFile
		exit(1)

	if not os.path.isdir(predFolder):
		print 'the folder for predicted bound matrix files does not exist: ', predFolder
		exit(1)

	if not os.path.isdir(nativefolder):
		print 'the folder for native distance matrix files does not exist: ', nativefolder
		exit(1)

	fh = open(proteinListFile, 'r')
	proteins = [ line.strip() for line in list(fh) ]
	fh.close()

	AccPerProtein = dict()
	accs = dict()
	for protein in proteins:
		predFile = os.path.join( predFolder, protein + fileSuffix ) 
		if not os.path.isfile(predFile):
			print 'the distance bound file does not exist: ', predFile
			exit(1)

		fh = open(predFile, 'rb')
		pred = cPickle.load(fh)
		fh.close()

		nativeFile = os.path.join(nativefolder, protein + '.atomDistMatrix.pkl' )
		if not os.path.isfile(nativeFile):
			print 'the native atomDistMatrix file does not exist: ', nativeFile
			exit(1)

		fh = open(nativeFile, 'rb')
		native = cPickle.load(fh)
		fh.close()		

                acc = DistanceUtils.EvaluateDistanceBoundAccuracy(pred[0], native, minSeqSep=minSeqSep)
		AccPerProtein[protein] = acc

		for k, v in acc.iteritems():
			if not accs.has_key(k):
				accs[k] = [ v ]
			else:
				accs[k].append(v)

	for k, v in accs.iteritems():
		accs[k] = np.average(v, axis=0)
		print 'average', k, str_display(accs[k])


	for k, v in AccPerProtein.iteritems():
		for apt, value in v.iteritems():
			print k, apt, str_display(value)


if __name__ == "__main__":
    	main(sys.argv[1:])
