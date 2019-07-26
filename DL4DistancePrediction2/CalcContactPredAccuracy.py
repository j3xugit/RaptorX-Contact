import numpy as np
import sys
import os

from ContactUtils import TopAccuracy
from ContactUtils import LoadContactMatrix

if __name__ == "__main__":

	if len(sys.argv) != 4:
    		print 'python CalcContactPredAccuracy.py pred_matrix_file distcb_matrix_file target'
		print '      Both matrix files are text format with L lines and each line has L columns where L is the protein sequence length'
    		exit(-1)

	predFile = sys.argv[1]
	distcbFile = sys.argv[2]
	target = sys.argv[3]

	pred = LoadContactMatrix(predFile)
	truth = LoadContactMatrix(distcbFile)

	accs = TopAccuracy(pred, truth)
	accsStr = [ str(a) for a in accs ]
	resultStr = target + ' ' + str(pred.shape[0]) + ' TopAcc '
	resultStr += (' '.join(accsStr) )
	print resultStr

	## the below method does not yield a better result
	##normalize prediction matrix

	"""
	colavgs = np.average(pred, axis=0)
	rowavgs = np.average(pred, axis=1)

	rowcolavgs = np.sqrt( np.outer(colavgs, rowavgs) )

	pred_normalized = np.divide( pred, rowcolavgs)
	accs = TopAccuracy(pred_normalized, truth)
	accsStr = [ str(a) for a in accs ]
	resultStr = target + ' ' + str(pred.shape[0]) + ' '
	resultStr += (' '.join(accsStr) )
	print resultStr
	"""
