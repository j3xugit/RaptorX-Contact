import numpy as np
import sys
import os

import ContactUtils

if __name__ == "__main__":

	if len(sys.argv) != 3:
    		print 'python CalcCASPContactPredAccuracy.py pred_CASP_file nativeDistMatrixFilePKL'
    		exit(-1)

	predFile = sys.argv[1]
	distFile = sys.argv[2]

	pred, target, sequence = ContactUtils.LoadContactMatrixInCASPFormat(predFile)

	accs = ContactUtils.EvaluateSingleCbCbContactPrediction(pred, distFile)
 
	accsStr = [ str(a) for a in accs ]
	resultStr = target + ' ' + str(pred.shape[0]) + ' TopAcc '
	resultStr += (' '.join(accsStr) )
	print resultStr


