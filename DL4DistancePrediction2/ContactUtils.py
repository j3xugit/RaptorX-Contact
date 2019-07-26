import os
import sys
import numpy as np
import cPickle

import config
import DataProcessor
import DistanceUtils
import Metrics

## load a contact matrix in text format. This matrix has L rows and L columns
def LoadContactMatrix(file=None):
    if file is None:
        print 'please provide a distance matrix file'
        exit(-1)

    if not os.path.isfile(file):
        print 'The distance matrix file does not exist: ', file
        exit(-1)

    content = np.genfromtxt(file, dtype=np.float32)

    return content

## this function loads contact prediction in CASP format and save it into a matrix
## here we assume the contact matrix is symmetric and 0 indicating that there is no predicted contact
def LoadContactMatrixInCASPFormat(filename):
	fh = open(filename, 'w')
 	content = [ line.strip() for line in list(fh) ]
    	fh.close()

    	if len(content) < 5:
        	print 'the input file contains fewer than 5 lines'
        	return None

    	##the first line must be "PFRMAT RR"
    	if content[0] != "PFRMAT RR":
        	print 'The first line of the input file is not PFRMAT RR'
        	return False

    	if content[1].startswith('TARGET') is not True:
        	print 'The second line of the input file shall start with TARGET.'
        	return None

    	targetName = content[1].split()[1]
    	sequence=""
    	probs = []

	for line in content[2:]:
    		if line.startswith('AUTHOR'):
        		author=line.split()[1]
			continue

    		if line.startswith('METHOD'):
			method=line.split()[1]
			continue

    		if line.startswith('MODEL'):
			modelNum=np.int32( line.split()[1] )
			assert modelNum==1, "currently only Model 1 is supported"

        	columns = line.split()
        	if len(columns) == 1:
            		sequence += columns[0]
        	elif len(columns) == 5:
            		indices = [ int(x) for x in columns[:2] ]
            		bounds = [ int(x) for x in columns[2:4] ]
            		prob = np.float32(columns[4])

            		if bounds[0] !=0 or bounds[1] !=8:
                		print 'wrong distance bound in the following line: '
                		print line
                		return None

            		if prob > 1 or prob <0 :
                		print 'The confidence score in the following line shall be between 0 and 1: '
                		print line
                		return None

            		if indices[0]<1 or indices[0]>len(sequence) or indices[1]<1 or indices[1]>len(sequence):
                		print 'The residue index in the following line is out of range: '
                		print line
                		return None

            		if indices[0] > indices[1]:
                		print 'The first index in a residue pair shall be smaller than the 2nd one:'
                		print line
                		return None

            		probs.append( indices + [ prob ] )

        	else:
            		print 'The following line in the input file has an incorrect format: '
            		print line
            		return None

	contactMatrix = np.zeros( (len(sequence), len(sequence)), dtype=np.float32)
	for p in probs:
		contactMatrix[ p[0], p[1] ] = p[2]
		contactMatrix[ p[1], p[0] ] = p[2]

	return contactMatrix, targetName, sequence


##this function assumes that the residue index starts from 1
def SaveContactMatrixInCASPFormat(target, sequence, contactMatrix, filename, probScaleFactor=config.ProbScaleFactor):

	if probScaleFactor == 1:
		contactMatrix2 = contactMatrix
	else:
		contactMatrix2 = np.power(contactMatrix, probScaleFactor)

	threshold4CASP = 0.05

	fh = open(filename, 'w')

        ##write the header information
        fh.write('PFRMAT RR\n')
        fh.write('TARGET ' + target + '\n')
        fh.write('AUTHOR RaptorX-Contact\n')
        fh.write('METHOD deep dilated residual networks (one variant of deep CNN). Consult jinboxu@gmail.com for details.\n')
        fh.write('MODEL 1\n')

        segmentLen = 50
        for i in range(0, len(sequence), segmentLen):
        	fh.write(sequence[i:i+segmentLen]+'\n')

	
        ##here we only consider the upper triangle, so this function cannot apply to nonsymmetri matrices such as CaCg, NO and HB matrices
        m = np.triu(contactMatrix2, 6)
        flatten_index = np.argsort( -1. * m, axis=None)
        real_index = np.unravel_index( flatten_index, contactMatrix2.shape )

	"""
	contactMapSize = np.prod(contactMatrix2.shape)
	if contactMapSize > 300000:
		threshold4CASP = 0.1
	"""

        ## numAllowedPairs is the maximum number of pairs allowed by CASP
	numAllowedPairs = 300000

	seqLen = len(sequence)
        maxNumPairs = len(real_index )

	## the minimum number of long-range residue pairs
        minNumLRPairs = min(3*seqLen, maxNumPairs)

	## the maximum number of allowed mediumm and short-range residue paris
        maxNumMSRPairs = maxNumPairs

        if maxNumPairs > numAllowedPairs:
                maxNumMRRPairs = max(0, maxNumPairs - minNumLRPairs)

        numPairs, numMSRPairs = 0, 0

        for i, j in zip(real_index[0], real_index[1]):
                if numPairs > numAllowedPairs:
                        break

		if i >= j:
			continue

                offset = abs(i - j)
                if offset < 6:
                        continue


		if numPairs>160000 and contactMatrix2[i, j]<threshold4CASP :
			continue

		"""
                if offset < 24 and numMSRPairs > maxNumMSRPairs:
                        continue
		"""

                numPairs += 1

                if offset < 24:
                        numMSRPairs += 1

                line = ' '.join( [ str(v) for v in [i+1, j+1, 0, 8] ] +[ "%.7f" % (contactMatrix2[i, j]) ] ) + '\n'
                fh.write(line)

	fh.write('END\n')
        fh.close()


## convert a dist prob matrix to a contact prob matrix
## labelOf8 is the cutoff label for contact
#def Distance2Contact(distProb, distLabelType):
def Distance2Contact(distProb, labelOf8=1):
	"""
        #labelOf8 = DistanceUtils.LabelsOfOneDistance(config.ContactDefinition, config.distCutoffs[distLabelType] )
        ContactProb = dict()

        for k, m in distProb.iteritems():
                ContactProb[k] = np.sum( m[:, :, :labelOf8], axis=2)
	"""
        contactProb = np.sum( distProb[:, :, :labelOf8], axis=2)

        return contactProb

##calculate MCC of a predicted contact matrix using a given score cutoff
##here we consider three cases: long-range contacts, long + medium-range contacts, long + medium- + short-range contacts
def CalcMCCF1(pred=None, truth=None, probCutoff=0.5, contactCutoff=8.0):
    if pred is None:
        print 'please provide a predicted contact matrix'
        exit(-1)

    if truth is None:
        print 'please provide a true distance matrix'
        exit(-1)

    assert pred.shape == truth.shape

    ## in case the matrix is not square, e.g., interfacial contact matrix
    seqLen = pred.shape[0]
    seqLen2 = pred.shape[1]

    pred_binary = (pred>probCutoff)
    truth_binary = ( 0<truth) & (truth<contactCutoff )
    pred_truth = pred_binary * 2 + truth_binary
    numPredicted = np.sum(pred_binary)
    numTruths = np.sum(truth_binary)
    #print "#predicted=", numPredicted, "#natives=", numTruths

    mask_LR = np.triu_indices(seqLen, 24, m=seqLen2)
    mask_MLR = np.triu_indices(seqLen, 12, m=seqLen2)
    mask_SMLR = np.triu_indices(seqLen, 6, m=seqLen2)


    metrics = []
    for mask in [ mask_LR, mask_MLR, mask_SMLR]:

        res = pred_truth[mask]
	total = res.shape[0]
	count = np.bincount(res, minlength=4)
	assert (total == np.sum(count) )

	## pred=0, truth=0	
	TN = count[0]

	## pred=0, truth=1
	FN = count[1]

	## pred=1, truth=0
	FP = count[2]

	## pred=1, truth=1
	TP = count[3]

	#print TP, FP, TN, FN

	MCC = Metrics.MCC(TP, FP, TN, FN)
	F1, precision, recall = Metrics.F1(TP, FP, TN, FN)

	metrics.extend ([MCC, TP, FP, TN, FN, F1, precision, recall])


    return np.array(metrics)


##this program outputs an array of contact prediction accuracy, arranged in the order of long-, medium-, long+medium- and short-range.
## for each range, the accuracy is calculated on the top L*ratio prediction where L is the sequence length.

## pred and truth are 2D matrices. Each entry in pred is a confidence score assigned to the corresponding residue pair indicating how likely this pair forms a contact
## truth is the ground truth distance matrix. The larger the distance, the more unlikely it is a contact. It is fine that one entry has value -1.
## in this distance matrix, only the entries with value between 0 and contactCutoff are treated as contacts.

def TopAccuracy(pred=None, truth=None, ratio=[1, 0.5, 0.2, 0.1], contactCutoff=8.0):
    if pred is None:
        print 'please provide a predicted contact matrix'
        exit(1)

    if truth is None:
        print 'please provide a true distance matrix'
        exit(1)

    assert pred.shape == truth.shape

    pred_truth = np.dstack( (pred, truth) )

    M1s = np.ones_like(truth, dtype = np.int8)
    mask_ER = np.triu(M1s, 48)
    mask_LR = np.triu(M1s, 24)
    mask_MLR = np.triu(M1s, 12)
    mask_SMLR = np.triu(M1s, 6)
    mask_MR = mask_MLR - mask_LR
    mask_SR = mask_SMLR - mask_MLR

    seqLen = pred.shape[0]

    accs = []
    for mask in [ mask_ER, mask_LR, mask_MR, mask_MLR, mask_SR]:

        res = pred_truth[mask.nonzero()]
	if res.size == 0:
                accs.extend( [0.0] * len(ratio) )
                continue

        res_sorted = res [ (-res[:,0]).argsort() ]

        for r in ratio:
            numTops = int(seqLen * r)
            numTops = min(numTops, res_sorted.shape[0] )
            topLabels = res_sorted[:numTops, 1]
            numCorrects = ( (0<topLabels) & (topLabels<contactCutoff ) ).sum()
            accuracy = numCorrects * 1./numTops
            accs.append(accuracy)

    return np.array(accs)

## Evaluate contact prediction for a single protein. predictedContactMatrix is a dictionary in which each key is a response.
## here response represents a pair of amino acid type, e.g., CbCb, CaCa, CgCg
def EvaluateSingleContactPrediction(predictedContactMatrix, nativefile):

        native = DataProcessor.LoadNativeDistMatrixFromFile(nativefile)
        accuracy = dict()

        for response in predictedContactMatrix.keys():
		if response.startswith('HB'):
                       	accuracy[response] = TopAccuracy(pred=predictedContactMatrix[response], truth=native[response], contactCutoff=config.MaxHBDistance)
		else:
                       	accuracy[response] = TopAccuracy(pred=predictedContactMatrix[response], truth=native[response])

	return accuracy

## pred  is a 2D contact matrix, each entry has a prob value
def EvaluateSingleCbCbContactPrediction(pred, nativefile):

        native = DataProcessor.LoadNativeDistMatrixFromFile(nativefile)
        accuracy = TopAccuracy(pred, truth=native['CbCb'])
	return accuracy


## predictedContactMatrices is a dictionary of contact matrices. Each key is a protein name
def EvaluateContactPredictions(predictedContactMatrices, nativefolder):

        ## load the ground truth
        allnatives = dict()
        for name in predictedContactMatrices.keys():
                allnatives[name] = DataProcessor.LoadNativeDistMatrix(name, nativefolder)
		"""
                if allnatives[name] is None:
                        print 'WARNING: cannot find the native distance matrix for protein ', name
		"""

        allaccuracy = dict()
        for name, results in predictedContactMatrices.iteritems():
		if not allnatives.has_key(name) or allnatives[name] is None:
			continue

                allaccuracy[name] = dict()
                for response in results.keys():
			##apt = Response2LabelName(response)
			apt = response
			if response.startswith('HB'):
                        	allaccuracy[name][response] = TopAccuracy(pred=predictedContactMatrices[name][response], truth=allnatives[name][apt], contactCutoff=config.MaxHBDistance)
			else:
                        	allaccuracy[name][response] = TopAccuracy(pred=predictedContactMatrices[name][response], truth=allnatives[name][apt])

        ## calculate average contact prediction accuracy
        allaccuracyByApt = dict()
        for name, results in allaccuracy.iteritems():
                for apt in results.keys():
                        if not allaccuracyByApt.has_key(apt):
                                allaccuracyByApt[apt] = [ results[apt] ]
                        else:
                                allaccuracyByApt[apt].append(results[apt])

	avgacc = dict()
        for apt in allaccuracyByApt.keys():
                avgaccuracyByApt = np.average(allaccuracyByApt[apt], axis=0)
		avgacc[apt] = avgaccuracyByApt

		"""
                print '******************Average contact prediction accuracy for atom pair type: ', apt, '*********************'
                print avgaccuracyByApt
		"""

	return avgacc, allaccuracy
