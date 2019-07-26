import numpy as np
import sys
import theano
import theano.tensor as T
from theano import printing
from theano.ifelse import ifelse
from numpy import random as rng
import os
import os.path
import time
import datetime
import gzip
import cPickle

modelFiles = [ 'ResNetv2-L93W112C-bn30.raptorx5.4738.pkl', 'ResNetv2-L91W212C-bn33.raptorx2.7536.pkl', 'ResNetv2-L41W2K212C-bn36.raptorx5.9833.pkl', 'ResNetv2-L57W212C-bn27.raptorx4.18440.pkl', 'ResNetv2-L71W212C-bn33.raptorx5.11577.pkl', 'ResNetv2-L73W112C-bn37.raptorx5.14491.pkl' ]
modelFiles = [ 'ResNetv2425CCbCb-L1D7L2D61Log101W1D7W2D2I1D26I2D6midLRbiasE20-pdb25-6767-train-11342.pkl', 'ResNetv2425CCbCb-L1D7L2D71Log101W1D7W2D2I1D26I2D6midLRbiasE19-pdb25-6767-train-18763.pkl', 'ResNetv2425CCbCb-L1D9L2D51Log101W1D7W2D2I1D26I2D6midLRbiasE19-pdb25-6767-train-28248.pkl']

import getopt
from ReadProteinFeatures import ReadFeatures
from RunDistancePredictor import PredictDistMatrix

##this function assumes that the residue index starts from 1
def SaveContactMatrixInCASPFormat(target, sequence, contactMatrix):

	filename = target + '.rr'
        fh = open(filename, 'w')

	##write the header information
	fh.write('PFRMAT RR\n')
	fh.write('TARGET ' + target + '\n')
	fh.write('AUTHOR RaptorX-Contact\n')
	fh.write('METHOD GCNN4DistPrediction\n')
	fh.write('MODEL 1\n')
	segmentLen = 50
	[ fh.write(sequence[i, i+segmentLen] for i in range(0, len(sequence), segmentLen) ]

        ##here we only consider the upper triangle, so this function cannot apply to CaCg and NO contact matrices
        m = np.triu(contactMatrix, 6)
        flatten_index = np.argsort( -1. * m, axis=None)
        real_index = np.unravel_index( flatten_index, contactMatrix.shape )

	seqLen = len(sequence)
	maxNumPairs = len(real_index )
	minNumLRPairs = min(3*seqLen, maxNumPairs)
	maxNumMSRPairs = maxNumPairs

	## 30000 is about the maximum number of pairs allowed by CASP
	if maxNumPairs > 30000:
		maxNumMRRPairs = max(0, maxNumPairs - minNumLRPairs)

	numPairs, numMSRPairs = 0, 0

        for i, j in real_index:
		offset = abs(i - j)
		if offset < 6:
			continue
		if numPairs > 30000:
			break
		
		if offset < 24 and numMSRPairs > maxNumMSRPairs:
			continue

		if offset < 24:
			numMSRPairs += 1
		numPairs += 1

                line = ' '.join( [ str(v) for v in [i+1, j+1, 0, 8, contactMatrix[i, j] ] ] ) + '\n'
                fh.write(line)
        fh.close()


def Usage():
    print 'python PredictDistMatrix4OneProtein.py target feature_folder1 [feature_folder2, feature_folder3, ...]'
    print '	This script assumes that all the Deep Learning models are stored in a folder $DL4DistancePredHome/Models/ where DL4DistancePredHome is an environmental variable'
    print '	target: the target name for which we would like to predict inter-residue distance and contacts'
    print '	feature_folders: specify at least one folders containing the protein features; each folder has an independent set of features for this target.'
    print '	the prediction result is stored in a file named after target.gcnn or target.CaCa.gcnn'

def main(argv):

    	featureDirs = None
    	target = None

    	if len(argv) < 2:
        	Usage()
        	exit(-1)

    	target = argv[0]
    	featureDirs = argv[1:]


    	print 'modelFiles=', modelFiles
    	print 'target=', target
    	print 'feature folders=', featureDirs

    	INSTALLDIR = os.getenv('DL4DistancePredHome')
    	if INSTALLDIR is None:
    		print 'please set the environment variable DL4DistancePredHome as the installation directory of the contact prediction program'
    		exit(-1)
	modelDir = os.path.join(INSTALLDIR, 'Models')
    	if not os.path.isdir(modelDir):
		print 'the model directory does not exist: ', modelDir
		exit(-1)

	sequence = None
    	proteins = []
    	for fDir in featureDirs:
		featureDir = fDir
        	if not featureDir.endswith('/'):
	    		featureDir += '/'
        	if not os.path.isdir(featureDir):
	    		print 'the following feature folder is invalid: ', featureDir
	    		exit(-1)
        	protein = ReadFeatures(p=target, DataSourceDir=featureDir)

		if sequence is None:
			sequence = protein['sequence']
		else:
			if sequence != protein['sequence']:
				print 'Error: inconsistent primary sequence among different features of the same protein ', target
				exit(-1)

		proteins.append(protein)

    	predFile = target + '.' + str(os.getpid()) + '.DistHBBetaFeatures.pkl'
    	savefh = open(predFile, 'wb')
    	cPickle.dump(proteins, savefh, protocol=cPickle.HIGHEST_PROTOCOL)
    	savefh.close()

	distPred, contPred, _ = PredictDistMatrix(modelFiles, predFile)

	assert len(contPred.keys() ) == 1

	##write code here to write the predicted contacts
	for apt, m in contPred.values()[0].iteritems():
		if apt == 'CbCb':
			savefilename = target + '.gcnn'
		else:
			savefilename = target + '.' + apt + '.gcnn'
		np.savetxt(savefilename, m, fmt='%.4f')

		if apt == 'CbCb':
			SaveContactMatrixInCASPFormat(target, sequence,m) 


"""
This program predicts the distance and contact matrix for a single protein using a combination of several models. 
One protein may have a few sets of input features
"""
if __name__ == "__main__":
    main(sys.argv[1:])
