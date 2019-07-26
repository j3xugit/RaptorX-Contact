import numpy as np
import sys
import theano
import theano.tensor as T
from numpy import random as rng
import os
import os.path
import time
import datetime
import gzip
import cPickle
import gc

import config
import DistanceUtils
import DataProcessor
import Model4DistancePrediction
import ContactUtils

from config import Response2LabelName, Response2LabelType
from utils import Compatible


import getopt

def Usage():
    print 'python RunDistancePredictor.py -m modelfiles -p predfiles [-d save_folder] [-g ground_truth_folder]'
    print '  A memory efficient implementation of RunDistancePredictor '
    print '-p: specify one or multiple files containing data to be predicted in PKL format, separated by semicolon'
    print '-m: specify one or multiple model files in PKL format, separated by semicolon'
    print '-d: specify where to save the result files (default current work directory). The result file of one protein is named after proteinName.predictedDistMatrix.pkl'
    print '	Each file saves a tuple of 6 items: proteinName, proteinSequence, predictedDistMatrixProb, predictedContactMatrix, labelWeightMatrix, and labelDistributionMatrix'
    print '-g: specify the ground truth folder containing all the native atom-level distance matrix files in PKL format'
    print '	when this option is provided, contact prediction accuracy will be calculated'


def PredictDistMatrix(modelFiles, predFiles, savefolder=None):
    	## load all the models from the files. Each file contains specification for one model.
	models = []
	for mFile in modelFiles:
    		fh = open(mFile, 'rb')
    		model = cPickle.load(fh)
    		fh.close()
		models.append(model)

	## check consistency among models. All the models shall have the same labelType for the same atom pair type
	labelTypes = dict()
	for model in models:
		for response in model['responses']:
			labelName = Response2LabelName(response)
			labelType = Response2LabelType(response)
			if not labelTypes.has_key(labelName):
				labelTypes[labelName] = labelType
			elif labelTypes[labelName] != labelType:
				print 'WARNING: at least two models have different label types for the same atom pair type.'
				exit(-1)
					

	allsequences = dict()

	##allresults shall be a nested dictionary, e.g, allresults[proteinName][response] = list of predicted_prob_matrices
	##We predict one prob_matrix from each model for each protein and each response
	## two different models may share some overlapping responses.

	allresults = dict()
	numModels = dict()
	for model, mfile in zip(models, modelFiles):
		if not model['network'] in config.allNetworks:

			print 'unsupported network architecture: ', model['network']
			exit(-1)

		distancePredictor, x, y, xmask, ymask, xem, labelList, weightList = Model4DistancePrediction.BuildModel(model, forTrain=False)

		inputVariables = [ x, y, xmask, ymask]
		if xem is not None:
			inputVariables.append(xem)

	  	pred_prob = distancePredictor.output_prob
        	predict = theano.function(inputVariables, pred_prob, on_unused_input='warn' )

		## set model parameter values
		if not Compatible(distancePredictor.params, model['paramValues']):
			print 'FATAL ERROR: the model type or network architecture is not compatible with the loaded parameter values in the model file: ', mfile
			exit(-1)

		[ p.set_value(v) for p, v in zip(distancePredictor.params, model['paramValues']) ]

		## We shall load these files for each model separately since each model may have different requirement of the data
		predData = DataProcessor.LoadDistanceFeatures(predFiles, modelSpecs = model, forTrainValidation=False)

		##make sure the input has the same number of features as the model. We do random check here to speed up
		rindex = np.random.randint(0, high=len(predData) )
		assert model['n_in_seq'] == predData[rindex]['seqFeatures'].shape[1]

		rindex = np.random.randint(0, high=len(predData) )
		assert model['n_in_matrix'] == predData[rindex]['matrixFeatures'].shape[2]

		if predData[0].has_key('embedFeatures'):
			rindex = np.random.randint(0, high=len(predData) )
			assert model['n_in_embed'] == predData[rindex]['embedFeatures'].shape[1]

		## check if all the proteins of the same name have exactly the same sequence
		for d in predData:
			if not allsequences.has_key(d['name']):
				allsequences[d['name']] = d['sequence']
			elif allsequences[d['name']] != d['sequence']:
				print 'Error: inconsistent primary sequence for the same protein in the protein feature files'
				exit(-1)
			
		## predSeqData and names are in the exactly the same order, so we know which data is for which protein	
		predSeqData, names = DataProcessor.SplitData2Batches(data=predData, numDataPoints=624, modelSpecs=model)
		print '#predData: ', len(predData), '#batches: ', len(predSeqData)

		for onebatch, names4onebatch in zip(predSeqData, names):
			input = onebatch[ : len(inputVariables) ]
			result = predict(*input)

			x1d, x2d, x1dmask, x2dmask = input[0:4]
			seqLens = x1d.shape[1] - x1dmask.shape[1] + np.sum(x1dmask, axis=1)
			maxSeqLen = x1d.shape[1]

			##result is a 4-d tensor. The last dimension is the concatenation of the predicted prob parameters for all responses in this model
			assert result.shape[3] == sum( [ config.responseProbDims[ Response2LabelType(res) ] for res in model['responses'] ] )

			## calculate the start and end positions of each response in the last dimension of result
			dims = [ config.responseProbDims[ Response2LabelType(res) ] for res in model['responses'] ]
                        endPositions = np.cumsum(dims)
                        startPositions =  endPositions - dims

			for name in names4onebatch:
				if not allresults.has_key(name):
					allresults[name]=dict() 
					numModels[name] =dict()

			## batchres is a batch of result, its ndim=4
			for response, start, end in zip(model['responses'], startPositions, endPositions):

				## the 1st dimension of batchres is batchSize, the 2nd and 3rd dimensions are contact/distance matrix sizes and the 4th is for the predicted probability parameters
				batchres = result[:, :, :, start:end ]


				## remove masked positions
				revised_batchres = [ probMatrix[ maxSeqLen-seqLen:, maxSeqLen-seqLen:, : ] for probMatrix, seqLen in zip(batchres, seqLens) ]

				for res4one, name in zip(revised_batchres, names4onebatch):
                                        if not allresults[name].has_key(response):
                                                allresults[name][response] = res4one
                                                numModels[name][response] = np.int32(1)
                                        else:
                                                ## here we save only sum to reduce memory consumption, which could be huge when many deep models are used to predict a large set of proteins
                                                allresults[name][response] +=  res4one
                                                numModels[name][response] += np.int32(1)


		del predict
		del predData
		del predSeqData
		gc.collect()


	## calculate the final result, which is the average of all the predictd prob matrices for the same protein and response
	finalresults = dict()
	for name, results in allresults.iteritems():
		if not finalresults.has_key(name):
			finalresults[name] = dict()

		## finalresults has 3 dimensions. 
		for response in results.keys():
			#finalresults[name][response] = np.average(allresults[name][response], axis=0)
			finalresults[name][response] = allresults[name][response]/numModels[name][response]

			##make the predicted distance prob matrices symmetric for some reponses. This also slightly improves accuracy.
			apt = Response2LabelName(response)
			if config.IsSymmetricAPT( apt ):
				finalresults[name][response] = (finalresults[name][response] + np.transpose(finalresults[name][response], (1, 0, 2) ) )/2.

	## collect the average label distributions and weight matrix. We collect all the matrices and then calculate their average.
	labelDistributions = dict()
	labelWeights = dict()
	for model in models:
		for response in model['responses']:
			apt = response
			if not labelDistributions.has_key(apt):
				labelDistributions[apt] = []
			if not labelWeights.has_key(apt):
				labelWeights[apt] = []

			labelDistributions[apt].append(model['labelRefProbs'][response])
			labelWeights[apt].append(model['weight4labels'][response])

	finalLabelDistributions = dict()
	finalLabelWeights = dict()

	for apt in labelDistributions.keys():
		finalLabelDistributions[apt] = np.average(labelDistributions[apt], axis=0)
	for apt in labelWeights.keys():
		finalLabelWeights[apt] = np.average(labelWeights[apt], axis=0)

	## convert the predicted distance probability matrix into a predicted contact matrix. 
	## Each predicted prob matrix has 3 dimensions while Each predicted contact matrix has 2 dimensions
	predictedContactMatrices = dict()
	from scipy.stats import norm
	for name, results in finalresults.iteritems():
		predictedContactMatrices[name] = dict()
		for response in results.keys():
			apt = Response2LabelName(response)
			labelType = Response2LabelType(response)

			if apt in config.allAtomPairTypes:
				if labelType.startswith('Discrete'):
					subType = labelType[len('Discrete'): ]
					labelOf8 = DistanceUtils.LabelsOfOneDistance(config.ContactDefinition, config.distCutoffs[subType])
					predictedContactMatrices[name][apt] =  np.sum( finalresults[name][response][:, :, :labelOf8], axis=2)
				elif labelType.startswith('Normal'):
					assert labelType.startswith('Normal1d2')
					normDistribution =  norm( loc=finalresults[name][response][:, :, 0], scale=finalresults[name][response][:,:,1])
					predictedContactMatrices[name][apt] =  normDistribution.cdf(config.ContactDefinition)
				elif labelType.startswith('LogNormal'):
					assert labelType.startswith('LogNormal1d2')
					normDistribution =  norm( loc=finalresults[name][response][:, :, 0], scale=finalresults[name][response][:,:,1])
					predictedContactMatrices[name][apt] =  normDistribution.cdf(np.log(config.ContactDefinition) )
				else:
					print 'unsupported label type in response: ', response
					exit(-1)

			elif apt in ['HB', 'Beta']:
				predictedContactMatrices[name][apt] =  finalresults[name][response][:, :, 0]
			else:
				print 'unsupported atom type in response: ', response
				exit(-1)


	##write all the results here
	## for each protein, we have a output file, which deposits a tuple like (predicted distance probability, labelWeight, RefProbs, predicted contact matrix, distLabelType, sequence)
        ## we store distLabelType for future use
	for name, results in finalresults.iteritems():

		savefilename = name + '.predictedDistMatrix.pkl'
		if savefolder is not None:
			savefilename = os.path.join(savefolder, savefilename)

		fh = open(savefilename, 'wb')
		cPickle.dump( (name, allsequences[name], results, predictedContactMatrices[name], finalLabelWeights, finalLabelDistributions), fh, protocol=cPickle.HIGHEST_PROTOCOL)
		fh.close()

	return finalresults, predictedContactMatrices, allsequences


def main(argv):

    	modelFiles = None
    	predFiles = None
	nativefolder = None
	savefolder = None

    	try:
       		opts, args = getopt.getopt(argv, "p:m:g:d:", ["predictfile=", "model=", "nativefolder=", "savefolder="])
        	#print opts, args
    	except getopt.GetoptError as err:
		print err
       		Usage()
        	exit(-1)


    	if len(opts) < 2:
        	Usage()
        	exit(-1)

    	for opt, arg in opts:
		if opt in ("-p", "--predictfile"):
           		predFiles = arg.split(';')
			for p in predFiles:
				if not os.path.isfile(p):
					print "input feature file does not exist: ", p
					exit(-1)

		elif opt in ("-m", "--model"):
            		modelFiles = arg.split(';')
			for m in modelFiles:
				if not os.path.isfile(m):
					print "model file does not exist: ", m
					exit(-1)

		elif opt in ("-d", "--savefolder"):
			savefolder = arg
			if not os.path.isdir(savefolder):
				print 'The specified folder for result save does not exist:', savefolder
				exit(-1)

		elif opt in ("-g", "--nativefolder"):
			nativefolder = arg
			if not os.path.isdir(nativefolder):
				print 'The specified folder does not exist or is not accessible:', nativefolder
				exit(-1)
		else:
            		print Usage()
            		exit(-1)

    	print 'modelFiles=', modelFiles
    	print 'predFiles=', predFiles
	print 'savefolder=', savefolder
	print 'nativefolder=', nativefolder

	assert len(modelFiles) > 0
	assert len(predFiles) > 0

	_, contPredictions, _ = PredictDistMatrix(modelFiles, predFiles, savefolder)

	if nativefolder is not None:
		#print 'nativeFolder=', nativefolder
		avgacc, allacc = ContactUtils.EvaluateContactPredictions(contPredictions, nativefolder)

		print 'Calculated contact prediction accuracy for %d proteins' % len(allacc.keys())
		for apt, acc in avgacc.iteritems():
			print 'average contact prediction accuracy for ', apt
			print acc

	
if __name__ == "__main__":
    main(sys.argv[1:])

