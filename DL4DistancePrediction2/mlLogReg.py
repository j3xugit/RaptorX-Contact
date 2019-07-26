"""
multi-layer neural network for classification using Theano.
"""

import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from Optimizers import AdaGrad, AdaDelta, SGDMomentum, GD
#from HF.hf import SequenceDataset, hf_optimizer
from LogReg import LogisticRegression as LogReg

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input
	self.n_in = n_in

        :type n_out: int
        :param n_out: number of hidden units
	self.n_out = n_out

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.n_in = n_in
        self.n_out = n_out


        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low = -numpy.sqrt(6. / (n_in + n_out)),
                    high = numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='HL_W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='HL_b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
        self.paramL1 = abs(self.W).sum() + abs(self.b).sum()
        self.paramL2 = (self.W**2).sum() + (self.b**2).sum()

        def errors(self, y):
            return T.sqrt(T.mean(T.pow(self.output - y, 2)))

# start-snippet-2
class MLLogReg(object):
    """Multi-Layer Logistic Classifier

    A multi-layer feedforward artificial neural network for classifier
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a linear regression layer (defined here by a ``LinearRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_out, n_hiddens=[]):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
	
	input has shape (batchSize, n_in)
	n_in is the number of input features
	n_out is the number of classes (or labels)

        :type n_hidden: int
        :param n_hidden: a tuple defining the number of hidden units at each hidden layer

        """

        self.input = input
        self.n_in = n_in
        self.n_hiddens = n_hiddens
        self.paramL1 =0
        self.paramL2 =0

        
        self.hlayers = []
        self.params = []

        output_in_last_layer = input
        n_out_in_last_layer = n_in

        for i in xrange(len(n_hiddens)):

            ## add one hidden layer
            hiddenLayer = HiddenLayer(
                rng = rng,
                input = output_in_last_layer,
                n_in = n_out_in_last_layer,
                n_out = n_hiddens[i],
                activation = T.tanh
            ) 
            	
	    self.paramL1 += hiddenLayer.paramL1
            self.paramL2 += hiddenLayer.paramL2
            self.params += hiddenLayer.params
            self.hlayers.append(hiddenLayer)

            output_in_last_layer = hiddenLayer.output
            n_out_in_last_layer = n_hiddens[i]


	## add the final logistic regression layer
        linLayer = LogReg(output_in_last_layer, n_out_in_last_layer, n_out)
	self.linLayer = linLayer
	self.paramL1 += linLayer.paramL1
        self.paramL2 += linLayer.paramL2
        self.params += linLayer.params

	self.pre_act = linLayer.pre_act
	self.p_y_given_x = linLayer.p_y_given_x
	self.y_pred = linLayer.y_pred

        self.output = self.y_pred
	self.n_out = n_out

    def negative_log_likelihood(self, y, sampleWeight=None):
        return self.linLayer.negative_log_likelihood(y, sampleWeight)

    def errors(self, y, sampleWeight=None):
	return self.linLayer.errors(y, sampleWeight)

    def loss(self, y, sampleWeight=None):
        return negative_log_likelihood(y, sampleWeight)



def testMLLogReg(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=2000, n_hiddens=[50,25], trainData=None, testData=None):

    ## generate some random train and test data
    trainX = numpy.random.uniform(0, 1, (10000, 20)).astype(numpy.float32)
    trainXsum = numpy.sum(trainX**2, axis=1)
    trainY = numpy.zeros((10000), dtype=numpy.int32 )
    numpy.putmask(trainY, trainXsum>5, 1)
    numpy.putmask(trainY, trainXsum>10, 2)
    numpy.putmask(trainY, trainXsum>15, 3)

    testX = numpy.random.uniform(0, 1, (10000, 20)).astype(numpy.float32)
    testXsum = numpy.sum(testX**2, axis=1)
    testY = numpy.zeros((10000), dtype=numpy.int32 )
    numpy.putmask(testY, testXsum>5, 1)
    numpy.putmask(testY, testXsum>10, 2)
    numpy.putmask(testY, testXsum>15, 3)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState()

    # construct the MLP class
    regressor = MLLogReg(rng, input=x, n_in=trainX.shape[1], n_hiddens=n_hiddens, n_out=4)

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        regressor.negative_log_likelihood(y)
        + L1_reg * regressor.paramL1
        + L2_reg * regressor.paramL2
    )
    # end-snippet-4

    gparams = [T.grad(cost, param) for param in regressor.params]
    param_shapes = [ param.shape.eval() for param in regressor.params ]
    #updates = SGDMomentum(regressor.params, gparams, 0.95, 0.001) 

    #train = theano.function( inputs=[x,y], outputs=[cost, regressor.errors(y)], updates=updates)
    test = theano.function( inputs=[x,y], outputs=regressor.errors(y))

    step = 10000
    tmpData0=[]
    tmpData1=[]
    for i in range(0,trainX.shape[0], step):
        tmpData0.append(trainX[i:i+step])
        tmpData1.append(trainY[i:i+step])
    trainSeqDataset = [tmpData0, tmpData1]

    tmpData0=[]
    tmpData1=[]
    for i in range(0,testX.shape[0], step):
        tmpData0.append(testX[i:i+step])
        tmpData1.append(testY[i:i+step])
    validSeqDataset = [tmpData0, tmpData1]

    gradient_dataset = SequenceDataset(trainSeqDataset, batch_size=None, number_batches=1)
    cg_dataset = SequenceDataset(trainSeqDataset, batch_size=None, number_batches=1)
    valid_dataset = SequenceDataset(validSeqDataset, batch_size=None, number_batches=1)

    hf_optimizer(regressor.params, [x,y], regressor.linLayer.pre_act, [cost, regressor.errors(y)]).train(gradient_dataset, cg_dataset, initial_lambda=1.0, preconditioner=True, num_updates=100, patience=10, validation=valid_dataset)

    print test(testX, testY)

if __name__ == '__main__':
    testMLLogReg()
