import numpy as np
import sys
import os

from ContactUtils import CalcMCCF1
from ContactUtils import LoadContactMatrix
import Metrics

def str_display(ls):
        if not isinstance(ls, (list, tuple, np.ndarray)):
                str_ls = '{0:.3f}'.format(ls)
                return str_ls

        str_ls = ['{0:.3f}'.format(v) for v in ls ]
        str_ls2 = '\t'.join(str_ls)
        return str_ls2


if __name__ == "__main__":

	if len(sys.argv) != 4:
    		print 'python BatchCalcMCCF1.py proteinListFile pred_matrix_dir distcb_matrix_dir'
		print '      The matrix files have text format with L lines and each line has L columns where L is the protein sequence length'
    		exit(-1)

	proteinListFile = sys.argv[1]
	predDir = sys.argv[2]
	truthDir = sys.argv[3]

	content=None
	with open(proteinListFile, 'r') as fh:
		content = [ line.strip() for line in list(fh) ]

	preds = {}
	truths = {}
	for protein in content:
		predFile = os.path.join(predDir, protein+'.gcnn')
		distcbFile = os.path.join(truthDir, protein+'.distcb')

		pred = LoadContactMatrix(predFile)
		truth = LoadContactMatrix(distcbFile)
		preds[protein] = pred
		truths[protein] = truth

	for prob in np.arange(20, 60, 1):
		#print "prob=", prob
		accs = []
		for protein in content:
			acc = CalcMCCF1(pred=preds[protein], truth=truths[protein], probCutoff=prob/100.)
			accs.append(acc)
		avgacc = np.average(accs, axis=0)

		resultStr = 'per-target avgMCCF1 at cutoff=' + str(prob) + ': ' + str_display(avgacc)
		print resultStr

		lrMCC = Metrics.MCC(avgacc[1], avgacc[2], avgacc[3], avgacc[4])
		lrF1, lrprecision, lrrecall = Metrics.F1(avgacc[1], avgacc[2], avgacc[3], avgacc[4])

		mrMCC = Metrics.MCC(avgacc[9], avgacc[10], avgacc[11], avgacc[12])
		mrF1, mrprecision, mrrecall = Metrics.F1(avgacc[9], avgacc[10], avgacc[11], avgacc[12])
		print 'per-pair avgMCCF1 at cutoff=' + str(prob) + ': ' + str_display([lrMCC, lrF1, lrprecision, lrrecall, mrMCC, mrF1, mrprecision, mrrecall])

