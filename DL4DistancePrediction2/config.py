import numpy as np
import theano.tensor as T

## this constant is used to scale up our predicted contact probability to maximize MCC and F1 values when p=0.5 is used as cutoff for binary contact classification.
## A probability value p is scaled to p^ProbScaleFactor, i.e., 0.4 is scaled to 0.5. 
## this scale factor is only used in saving the predicted contact matrix into a CASP submission file (i.e., in generating CASP.rr file)
## in CASP12, we did not do this and our MCC and F1 score are not the best (when p=0.5 is used as cutoff), although I do think such a scale-up is meaningless.
## this scale constant depends on the weight factor we used in calculating loss function, so whenever the weight factor changes, this constant shall be adjusted.
ProbScaleFactor = np.log(0.5)/np.log(0.4)

## ResNet2DV21 and ResNet2DV22 are added on March 8, 2018
## ResNet2DV21 is the same as ResNet2D. ResNet2DV23 is recommended.
## ResNet2DV23 is almost same as ResNet2D except that the former has removed unused batch normalization layers (and parameters)
## ResNet2DV22 differs from ResNet2DV21 in that the former has two batch norm layers in each residual block while the latter has only one
## ResNet2DV22 seems to be better than ResNet2DV21, but maybe this depends on training algorithm and learning rate
allNetworks = ['ResNet2D', 'ResNet2DV21', 'ResNet2DV22', 'ResNet2DV23', 'DilatedResNet2D']


allDistLabelTypes = [ ('Discrete' + label) for label in ['52C', '36C','34CPlus', '34C', '25CPlus', '25C', '14CPlus', '14C', '13CPlus', '13C', '12CPlus', '12C', '3CPlus', '3C', '2CPlus', '2C' ] ]
allLabelTypes = allDistLabelTypes + ['Normal', 'LogNormal']

allAtomPairTypes = ['CbCb', 'CaCa', 'CgCg', 'CaCg', 'NO']
allLabelNames = allAtomPairTypes + ['HB', 'Beta']
symAtomPairTypes = ['CbCb', 'CaCa', 'CgCg', 'Beta']

def ParseAtomPairTypes(aptStr):
	if aptStr.upper() == 'All'.upper():
        	apts = allAtomPairTypes
        else:
                apts = aptStr.split('+')
	return apts


def IsSymmetricAPT( apt ):
	return  ( apt in set(symAtomPairTypes) )

topRatios = dict()
for apt in allAtomPairTypes:
	topRatios[apt] = 0.5
topRatios['HB'] = 0.1
topRatios['Beta'] = 0.1

allAlgorithms = ['SGDM', 'SGDM2', 'Adam', 'SGNA', 'AdamW', 'AdamWAMS', 'AMSGrad']
#allEmbeddingModes = ['SeqOnly', 'Seq+SS', 'Profile', 'OuterCat']
allEmbeddingModes = ['SeqOnly', 'Seq+SS', 'OuterCat']


## In a distance matrix, -1 represents an invalid distance (i.e, at least one residue has no valid 3D coodinates in PDB file) and a positive value represents a valid distance
## in the beta-pairing (Beta) or hydrogen-bonding (HB) matrix, we still use -1 to indicate that there is no valid distance between two Cbeta atoms
## we also use a value 100 + real_distance to indicate that one entry does not form a beta pairing or hydrogen bond but has distance=real_distance
## when one entry in a Beta or HB matrix forms a beta pairing or hydrogen bond, this entry contains the real distance of the Cbeta atoms.
## the maximum Cbeta distance of two residues forming a beta pair is approximately 8 Angstrom
## the maximum Cbeta distance of two residues forming a hydrogen bond is slightly more than 9 Angstrom

MaxBetaDistance = 8.0
MaxHBDistance = 9.5

## a response has format such as 25CPlus, 13CPlus, 12C, 12CPlus 3C, 3CPlus. 
## When Plus is used, the nonexisting distance -1 is separated from the maximum distance bin
## otherwise it is merged with the maximum distance bin

distCutoffs = {}

distCutoffs['52C'] = np.array ( [0] + np.linspace(4.0, 16.5, num=51).tolist()  ).astype(np.float32)
distCutoffs['36C'] = np.array ( [0] + np.linspace(4.15, 16.4, num=35).tolist()  ).astype(np.float32)

distCutoffs['34CPlus'] = np.array ( [0] + np.linspace(4.0, 20.0, num=33).tolist()  ).astype(np.float32)
distCutoffs['34C'] = distCutoffs['34CPlus']

distCutoffs['25CPlus'] = np.array ( [0] + np.linspace(4.5, 16.0, num=24).tolist()  ).astype(np.float32)
distCutoffs['25C'] = distCutoffs['25CPlus']

distCutoffs['14CPlus'] = np.array( [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] ).astype(np.float32)
distCutoffs['14C'] =  distCutoffs['14CPlus']

distCutoffs['13CPlus'] = np.array( [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] ).astype(np.float32)
distCutoffs['13C'] =  distCutoffs['13CPlus']

distCutoffs['12CPlus'] = np.array( [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ] ).astype(np.float32)
distCutoffs['12C'] = distCutoffs['12CPlus']

distCutoffs['3CPlus'] = np.array( [0, 8, 15] ).astype(np.float32)
distCutoffs['3C'] = distCutoffs['3CPlus']

distCutoffs['2C'] = np.array( [0, 8 ] ).astype(np.float32)
distCutoffs['2CPlus'] = distCutoffs['2C']

distCutoffs_HB = {}
distCutoffs_HB['2C'] = np.array( [0, MaxHBDistance] ).astype(np.float32)
distCutoffs_HB['2CPlus'] = distCutoffs_HB['2C']

#the true repsonse is the combination of one element in allLabelNames and one element in allLabelTypes
def Response2LabelType(response):
        return response.split('_')[1]

def Response2LabelName(response):
        return response.split('_')[0]

def ParseResponse(response):
	return response.split('_')

## the number of dimensions for a response variable when represented as a predicted value
## currently only 1d Normal or LogNormal is implemented. 
## to support the 2d Normal or LogNormal, need to check other places 
responseValueDims = dict()
responseValueDims['Normal'] = 1
responseValueDims['Normal2d'] = 2
responseValueDims['Normal2d2'] = 2
responseValueDims['Normal2d4'] = 2

responseValueDims['LogNormal'] = 1
responseValueDims['LogNormal2d'] = 2
responseValueDims['LogNormal2d2'] = 2
responseValueDims['LogNormal2d4'] = 2

for distLabelType in allDistLabelTypes:
	responseValueDims[ distLabelType ] = 1

## the number of paramters for the probability distribution function of a response
responseProbDims = dict()
responseProbDims['Normal']=2
responseProbDims['Normal2d']=5
responseProbDims['Normal2d2']=2
responseProbDims['Normal2d4']=4

responseProbDims['LogNormal']=2
responseProbDims['LogNormal2d']=5
responseProbDims['LogNormal2d2']=2
responseProbDims['LogNormal2d4']=4

for distLabelType in allDistLabelTypes:
	if distLabelType.endswith('C'):
		responseProbDims[ distLabelType ] = np.int32(distLabelType[len('Discrete'): -1] )
	elif distLabelType.endswith('CPlus'):
		responseProbDims[ distLabelType ] = np.int32(distLabelType[len('Discrete'): -5] ) + 1
	else:
		print 'unsupported distance label type: ', distLabelType
		exit(1)

## weight for different ranges: long-range, medium-range, short-range and near-range residue pairs
RangeBoundaries = [ 24, 12, 6, 2]
numRanges = len(RangeBoundaries)

def GetRangeIndex(offset):
        if offset < RangeBoundaries[-1]:
                return -1

        rangeIndex = 0
        for l in range(numRanges):
                if offset >= RangeBoundaries[l]:
                        break
                else:
                        rangeIndex += 1
        return rangeIndex

weight4range = np.array([ 3., 2.5, 1., 0.5]).reshape((-1,1)).astype(np.float32)


##weight for 3 distance intervals: 0-8, 8-15, >15 or -1
##each row is the distance weight for one specific range. In total there are 4 ranges, ordered from long-, to medium, to short and to near range.
##for example, in [17, 4, 1], 17 is the weight for 0-8, 4 for 8-15 and 1 for >15 or -1
weight43C = dict()
weight43C['low'] = np.array( [ [17, 4, 1], [5, 2, 1], [2.5, 0.6, 1], [0.2, 0.3, 1] ] ).astype(np.float32)
weight43C['mid'] = np.array( [ [20.5, 5.4, 1], [5.4, 1.89, 1], [2.9, 0.7, 1], [0.2, 0.3, 1] ] ).astype(np.float32)
weight43C['high']= np.array( [ [23, 6, 1], [6, 2.5 ,1], [3, 1, 1] ,[0.2, 0.3, 1] ] ).astype(np.float32)
weight43C['veryhigh'] = np.array( [ [25, 6, 1], [7, 2.5 ,1], [3, 1, 1], [0.2, 0.3, 1] ] ).astype(np.float32)
weight43C['exhigh']  =np.array( [ [28, 6, 1], [8, 2.5 ,1], [4, 1, 1], [0.2, 0.3, 1] ] ).astype(np.float32)

# weight for Beta-pairing, only two labels, 0 for positive and 1 for negative
weight4Beta2C = np.array( [ [360, 1], [70, 1], [50, 1], [120, 1] ] ).astype(np.float32) 

# weight for hydrogen-bonding, only two labels, 0 for positive and 1 for negative
weight4HB2C = np.array( [ [600., 1], [120., 1], [90., 1], [5., 1] ] ).astype(np.float32) 

## the distance cutoff for Cbeta-Cbeta contact definition
ContactDefinition = 8.001

## when the distance between two atoms is beyond this cutoff, we assume they have no interaction at all
InteractionLimit = 15.001

def InitializeModelSpecs():
	modelSpecs = dict()
        modelSpecs['trainFile'] = None
        modelSpecs['validFile'] = None
        modelSpecs['predFile'] = None
        modelSpecs['checkpointFile'] = None

        modelSpecs['network'] = 'ResNet2D'
        modelSpecs['responseStr'] = 'CbCb:25C'
	modelSpecs['responses'] = ['CbCb_Discrete25C']
	modelSpecs['w4responses'] = [ 1. ]
	modelSpecs['topRatios'] = [ topRatios['CbCb'] ]

        modelSpecs['algorithm'] = 'Adam'
        modelSpecs['numEpochs'] = [ 19, 2 ]
        modelSpecs['lrs'] = [np.float32(0.0002), np.float32(0.0002)/10 ]

	modelSpecs['algorithm4var'] = 'Adam'
        modelSpecs['numEpochs4var'] = modelSpecs['numEpochs']
	modelSpecs['lrs4var'] = modelSpecs['lrs']

	modelSpecs['algStr'] = 'Adam:21+0.00022'

        modelSpecs['validation_frequency'] = 100
        modelSpecs['patience'] = 5


        ##default number of hidden units at 1d convolutional layer
        modelSpecs['conv1d_hiddens'] = [30, 35, 40, 45]
        modelSpecs['conv1d_repeats'] = [ 0,  0,  0,  0]
	modelSpecs['conv1d_hwsz'] = 7

        ## the number of hidden units at 2d convolutional layer
        modelSpecs['conv2d_hiddens'] = [50, 55, 60, 65, 70, 75]
        modelSpecs['conv2d_repeats'] = [4,  4,  4,  4,  4,  4 ]
	modelSpecs['conv2d_hwszs'] =   [1,  1,  1,  1,  1,  1 ]
	modelSpecs['conv2d_dilations'] = [1, 1, 2,  4,  2,  1 ]

        ## for the logistic regression at the final stage
        modelSpecs['logreg_hiddens'] = [ 80 ]

        modelSpecs['halfWinSize_seq'] = 7
        modelSpecs['halfWinSize_matrix'] = 2
        modelSpecs['activation'] = T.nnet.relu

        modelSpecs['seq2matrixMode'] = {}
        modelSpecs['seq2matrixMode']['SeqOnly' ] = [ 4, 6, 12 ]
        modelSpecs['seq2matrixMode']['OuterCat' ] = [ 70, 35 ]

        modelSpecs['L2reg'] = 0.0001

        modelSpecs['minibatchSize'] = 60000
        modelSpecs['maxbatchSize'] = 350*350

	## input features
	modelSpecs['UseSequentialFeatures'] = True
	modelSpecs['UseSS'] = True
	modelSpecs['UseACC'] = True
	modelSpecs['UsePSSM'] = True
	modelSpecs['UseDisorder'] = False
	modelSpecs['UseCCM'] = True

	##OtherPairs include mutual information and contact potential
	modelSpecs['UseOtherPairs'] = True

        modelSpecs['UsePriorDistancePotential'] = False
        modelSpecs['UsePSICOV'] = False

	## bias added for long-range prediction
        modelSpecs['LRbias'] = 'mid'

        ## by All, we consider all-range residue pairs including those pairs (i, j) where abs(i-j)<6
        modelSpecs['rangeMode'] = 'All'

        modelSpecs['batchNorm'] = True
        modelSpecs['UseSampleWeight'] = True
        modelSpecs['SeparateTrainByRange'] = False

	return modelSpecs

def SelectCG(AA):
	#print 'AA=', AA
        a2 = 'cg'
        if AA == 'V' or AA == 'I':
                a2 = 'cg1'
        elif AA == 'T':
                a2 = 'cg2'
	elif AA == 'S':
		a2 = 'og'
	elif AA == 'C':
		a2 = 'sg'
        elif AA == 'A':
                a2 = 'cb'
        elif AA == 'G':
                a2 = 'ca'
        return a2

def SelectAtomPair(sequence, i, j, atomPairType):

        if atomPairType == 'CaCa':
                return 'ca', 'ca'
        if atomPairType == 'NO':
                return 'N', 'O'

        if atomPairType == 'CbCb':
                a1, a2 = 'cb', 'cb'
                if sequence[i] == 'G':
                        a1 = 'ca'
                if sequence[j] == 'G':
                        a2 = 'ca'
                return a1, a2

        if atomPairType == 'CaCg':
                a1 = 'ca'
                a2 = SelectCG(sequence[j].upper())
                return a1, a2

        if atomPairType == 'CgCg':
                a1 = SelectCG(sequence[i].upper())
                a2 = SelectCG(sequence[j].upper())
                return a1, a2


def EmbeddingUsed(modelSpecs):
	if not modelSpecs.has_key('seq2matrixMode'):
		return False
	return any( k in modelSpecs['seq2matrixMode'] for k in ('SeqOnly', 'Seq+SS') )

def InTPLMemorySaveMode(modelSpecs):
	if not modelSpecs.has_key('TPLMemorySave'):
		return False
	return modelSpecs['TPLMemorySave']

## encoding 20 amino acids, only used to represent a primary sequence as a L*20 matrix
AAOrders = { 'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E' : 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19 }
AAs = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'] 
AAVectors = np.zeros((26,20)).astype(np.int32)

for aa in AAs:
	index = ord(aa) - ord('A') 
	AAVectors[index][ AAOrders[aa] ] = 1

## conduct one-hot encoding of a protein sequence, which consists of a bunch of amino acids. Each amino acid is an element in AAs
def SeqOneHotEncoding(sequence):
	seq2int = (np.array(map(ord, sequence)) - ord('A') ).astype(np.int32)
	return AAVectors[seq2int]

AA3LetterCode21LetterCode = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
