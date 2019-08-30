import os
import sys
import cPickle
import numpy as np
import copy

import config

## load the predicted distance matrix
def LoadRawDistProbFile(file=None):
        if file is None:
                print 'please provide a raw distance probability distribution file with suffix .predictedDistMatrix.pkl'
                exit(1)

        if not os.path.isfile(file):
                print 'The specified file does not exist: ', file
                exit(1)

        fh = open(file, 'rb')
        content = cPickle.load(fh)
        fh.close()

        return content

## bound is a dict() and bound['CbCb'] is a matrix with dimension L*L*10
## native is a dict(), native['CbCb'] is a 2D distance matrix
def EvaluateDistanceBoundAccuracy(bound, native, minSeqSep=12, epsilon=0.000001):

	querySeq = bound['seq']
	nativeSeq = native['seq4matrix']

	pos = nativeSeq.find(querySeq)
	if pos < 0:
		print 'ERROR: the query sequence in predicted bound is not a substring of the native sequence'
		print 'querySeq: ', querySeq
		print 'nativeSeq: ', nativeSeq
		exit(1)

	start = pos
	end = pos + len(querySeq)

	accs = dict()
	for apt in bound.keys():
		if apt == 'seq':
			continue

		pred = bound[apt][:,:,0]
		truth = native[apt][start:end, start:end]

		## truth_valid is a flag matrix to indicate one native distance is valid or not
		## one entry in truth is invalid if it is negative. 
		truth_valid = np.ones_like(truth)
		np.putmask(truth_valid, truth < 0.1,  0)

		## pred_valid is a flag matrix to indicate if an entry in pred is valid or not. An estimated distance is valid if it is between 0 and 15.
		pred_valid = np.zeros_like(pred)
		np.putmask(pred_valid, pred>0, 1)
		np.putmask(pred_valid, pred>15, 0)

		## exclude short-range residue pairs. Only consider those pairs of residues with sequence separation >=12
		for offset in range(0, minSeqSep):
			np.fill_diagonal(truth_valid[:-offset, offset:], 0)
			np.fill_diagonal(pred_valid[:-offset, offset:], 0)

		for offset in range(1, minSeqSep):
			np.fill_diagonal(truth_valid[offset:, :-offset], 0)
			np.fill_diagonal(pred_valid[offset:, :-offset], 0)

		## calculate difference between pred and truth
		diff = abs(pred - truth)
		diff_valid = truth_valid * pred_valid

		## absolute error
		abs_error = np.sum( diff * diff_valid) /(epsilon + np.sum(diff_valid) )

		## relative error
		avg_dist = abs(pred + truth)/2 + epsilon
		rel_diff = diff / avg_dist
		rel_error = np.sum(rel_diff * diff_valid) /(epsilon + np.sum(diff_valid) )

		## precision = the percentage of valid entries in pred with corresponding native distance between 0 and 15
		## recall = the percentage of entries<=15 in truth with a valid pred, i.e., the predicted distance falls into [0, 15]
		truth_valid_15 = copy.deepcopy(truth_valid)
		np.putmask(truth_valid_15, truth>15, 0)
		TP = np.sum(pred_valid * truth_valid_15)
		precision = TP*1./(epsilon + np.sum(pred_valid * truth_valid) )
		recall = TP*1./(epsilon + np.sum(truth_valid_15) )
		F1 = 2*precision*recall/(precision+recall+epsilon)

		## calculate similarity score using a metric similar to GDT
		sim = np.zeros_like(diff)
		np.putmask(sim, diff < 8, 1./8)
		np.putmask(sim, diff < 4, 1./4)
		np.putmask(sim, diff < 2, 1./2)
		np.putmask(sim, diff < 1, 1.)
		GDT = np.sum(sim * diff_valid) /(epsilon + np.sum(diff_valid))


		accs[apt] = [abs_error, rel_error, precision, recall, F1, GDT]

	return accs

"""
originalProb is the originally predicted atomic distance prob matrix. It has shape [L, L, numLabels]
labelWeight is the weight assigned to labels in training
refProb is the background probability of labels derived from training data
The rows of RefProb and labelWeight correspond to 5 different ranges
"""

def FixDistProb(originalProb, labelWeight, refProb):

        newRefProb1 = np.multiply(labelWeight[0: config.numRanges], refProb[0: config.numRanges])
        newRefProb1_sum = np.sum(newRefProb1, axis=1, keepdims=True)
        newRefProb = newRefProb1 / newRefProb1_sum

        size = originalProb.shape
        fixedProb = np.zeros_like(originalProb)

        for i in range(size[0]):
                for j in range(size[1]):
                        offset = abs(i-j)
			rangeIndex = config.GetRangeIndex(offset)
			if rangeIndex < 0:
				continue

                        tmpProb = originalProb[i, j] * refProb[rangeIndex] / newRefProb[rangeIndex]
                        fixedProb[i, j] = tmpProb / np.sum(tmpProb)

        return fixedProb

## this function merges a distance prob matrix for x bins to a new prob matrix for y distance bins where y < x 
## the code needs some revision if you want to deal with dist label type ending with 'Plus'
def MergeDistanceBins(srcProbMatrix, srcDistCutoff, dstDistCutoff):
	assert ( len(dstDistCutoff) < len(srcDistCutoff) )

	## find the index positions of each distance boundary of dstDistCutoff in srcDistCutoff
	positions = [ LabelsOfOneDistance(d+0.00001, srcDistCutoff) for d in dstDistCutoff ]

	matrixList = []
	for i in xrange(len(positions)-1 ):
		start = positions[i]
		end = positions[i+1]

		m = np.sum(srcProbMatrix[:,:,start:end], axis=-1, keepdims=True)
		matrixList.append(m)

	matrixList.append( np.sum(srcProbMatrix[:,:, positions[-1]: ], axis=-1, keepdims=True) )

	## concatenate matrixList into a new matrix
	dstProbMatrix = np.concatenate(matrixList, axis=-1)

	return dstProbMatrix

## if bins is None, then use subType to determine the cutoffs
## otherwise, directly use bins as the cutoff
def DiscretizeDistMatrix(distm, bins=None, invalidDistanceSeparated=False):

	assert (bins is not None)

	result = np.digitize(distm, bins) - 1
	if invalidDistanceSeparated:
		## -1 and the maximum distance bin are separated
		np.putmask(result, result == -1, len(bins) )
		labels = range( len(bins) + 1 )
	else:
		## -1 and the maximum distance bin are merged into a single bin
		np.putmask(result, result == -1, len(bins) -1 )
		labels = range( len(bins) )

	return result.astype(np.int32), np.int32(labels), bins

## calculate the natural logarithm of a matrix
def LogDistMatrix(distm):
	tmpMatrix = copy(distm)
	np.putmask(tmpMatrix, distm < 1/np.e, 1/np.e)
	return np.log(tmpMatrix)

## this function calculates the joint label probability distribution at different ranges: extra-, long-, medium-, short- and near-range
## data is a list of matrices with shape (L, L, 2) where L is the sequence length. 
## [:, :, 0] is for target protein and [:, :, 1] is for template information
## numLabels is the number of different labels in the matrices
## this function returns 3 values: joint probability, marginal probability for target protein, marginal probability for template

def CalcJointLabelProb(data=None, numLabels=52):
	pass

## this function calculates the label probability distribution at four different ranges: long-, medium-, short- and near-range
## data is a list of label matrices, numLabels is the number of possible labels
def CalcLabelProb(data=None, numLabels=26, eps=np.finfo(np.float32).eps):

	freqs = [ ]
        for separation in config.RangeBoundaries:
		#print 'separation=', separation
		freq = []
        	for m in data:
                        index = np.triu_indices(m.shape[0], separation)
                        values = m[index]
                        res = np.bincount(values, minlength=numLabels )
                        freq.append(res)
		freqs.append(np.sum(freq, axis=0) )

        count = np.array(freqs)
	#print count.shape
	#print count

        ## the order of subtraction cannot be changed
        ## count[0], [1], [2], [3], [4] are for extra long-, long-, medium-, short- and near-range residue pairs, respectively
	for i in range(count.shape[0]-1, 0, -1):
		count[i] -= count[i-1]

        frequency = [ c/(eps + np.sum(c) ) for c in count ]
        return np.array(frequency)


## calculate label distribution from a list of distance matrices
## when invalidDistanceSeparated is True, it means that the invalid distance (represented by -1) is treated as an independent distance bin
def CalcDistProb(data=None, bins=None, invalidDistanceSeparated=False):

	labelMatrices = [ ]
	for distm in data:
                #m, _, _ = DiscretizeDistMatrix(distm, subType=subType)
                m, _, _ = DiscretizeDistMatrix(distm, bins=bins, invalidDistanceSeparated=invalidDistanceSeparated)
		labelMatrices.append(m)

	## need fix here
	#probs = CalcLabelProb( labelMatrices, config.responseProbDims['Discrete' + subType] )
	if invalidDistanceSeparated:
		probs = CalcLabelProb( labelMatrices, len(bins) + 1 )
	else:
		probs = CalcLabelProb( labelMatrices, len(bins) )

        return probs

## d needs to be positive, cannot be -1
## cutoffs is the distance boundary array
## return the largest index position such that cutoffs[position] <= d, i.e.,  d< cutoffs[position+1]
def LabelsOfOneDistance(d, cutoffs):
	result = np.digitize(np.array([d]), cutoffs) - 1
	return np.int16(result[0])

##this function calculates the weight for the labels from the initial weight assigned to '3C'
## weight43C is a 4*3 matrix, the rows corresponding to long-, medium-, short- and near-range and the cols corresponding to three distance bins (0-8, 8-15, >15)
## ref_prob is the background label distribution, being a matrix of 4 * numLabels where numLabels = len(distCutoffs) or plus 1
## this function returns a 4 * numLabels matrix with each entry being the weight assigned to a specific label

def CalcLabelWeight(weight43C, ref_prob, distCutoffs):
	assert len(distCutoffs) == ref_prob.shape[1] or len(distCutoffs)+1 == ref_prob.shape[1]

	labelOf8 = LabelsOfOneDistance(config.ContactDefinition, distCutoffs )
	labelOf15 = LabelsOfOneDistance(config.InteractionLimit, distCutoffs )

	weight = np.ones_like(ref_prob, dtype=weight43C.dtype)
	for w, w3C, rp in zip(weight, weight43C, ref_prob):
		avg_08 = np.average(rp[0: labelOf8] )
		avg_815 = np.average( rp[labelOf8 : labelOf15] )
		avg_15 = np.average( rp[labelOf15: ] )
		w[0: labelOf8 ] = w3C[0] * avg_08 / rp[0: labelOf8]
		w[labelOf8 : labelOf15 ] = w3C[1] * avg_815 / rp[labelOf8 : labelOf15]
		w[labelOf15:] = w3C[2] * avg_15 / rp[labelOf15:]

	return weight

def TestDiscretize():
	
	##create a random matrix
	dm = np.random.rand(8, 8) * 20
	#dm[0, 2] = -1
	#dm[3, 5] = -1

	print dm
	#print DiscretizeDistMatrix(dm, '25CPlus')
	res, labels, bins = DiscretizeDistMatrix(dm, config.distCutoffs['25C'])
	print res
	print labels
	print bins

	print LabelsOfOneDistance(config.ContactDefinition, bins)
	print LabelsOfOneDistance(config.InteractionLimit, bins)


if __name__ == "__main__":
	
	TestDiscretize()

