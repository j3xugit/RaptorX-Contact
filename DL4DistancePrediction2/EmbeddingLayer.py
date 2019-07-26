import theano
import theano.tensor as T
import numpy as np
import sys
import os


class EmbeddingLayer(object):
    
    def __init__(self, input, n_in, n_out):
	## input has shape (batchSize, seqLen, n_in)
	## input shall be a binary tensor, each row has only one 1
	
	self.n_in = n_in
	self.n_out = n_out
	self.input = input

	value_bound = np.sqrt(6./(n_in * n_in  + n_out))
	W_values = np.asarray(np.random.uniform(low = - value_bound, high = value_bound, size=(n_in, n_in, n_out)), dtype=theano.config.floatX)
	self.W = theano.shared (value = W_values, name = 'EmbeddingLayer_W', borrow=True)


	## out1 shall have shape (batchSize, seqLen, n_in, n_out)
	out1 =  T.tensordot(input, self.W, axes=1)

	##out2 has shape(batchSize, n_out, seqLen, n_in)
	out2 = out1.dimshuffle(0, 3, 1, 2)

	##input2 has shape(batchSize, n_in, seqLen)
	input2 = input.dimshuffle(0,2,1)

	##out3 shall have shape (batchSize, n_out, seqLen, seqLen)
	out3 = T.batched_tensordot(out2, input2, axes=1)

	##output has shape (batchSize, seqLen, seqLen, n_out)
	self.output = out3.dimshuffle(0, 2, 3, 1)

	self.params = [self.W]
	self.paramL1 = abs(self.W).sum()
	self.paramL2 = (self.W**2).sum()
	##self.pcenters = (self.W.sum(axis=[0, 1])**2 ).sum()
	self.pcenters = (self.W.mean(axis=[0, 1])**2 ).sum()

class MetaEmbeddingLayer(object):
    def __init__(self, input, n_in, n_out):

	batchSize, seqLen, _ = input.shape

	import collections
	if isinstance(n_out, collections.Sequence):
            LRembedLayer = EmbeddingLayer(input, n_in, n_out[2])
            MRembedLayer = EmbeddingLayer(input, n_in, n_out[1])
            SRembedLayer = EmbeddingLayer(input, n_in, n_out[0])
	    n_out_max = max(n_out)
	else:
            LRembedLayer = EmbeddingLayer(input, n_in, n_out)
            MRembedLayer = EmbeddingLayer(input, n_in, n_out)
            SRembedLayer = EmbeddingLayer(input, n_in, n_out)
	    n_out_max = n_out

	self.layers = [ LRembedLayer, MRembedLayer, SRembedLayer]

	M1s = T.ones( (seqLen, seqLen) )
	Sep24Mat = T.triu(M1s, 24) + T.tril(M1s, -24)
	Sep12Mat = T.triu(M1s, 12) + T.tril(M1s, -12)
	Sep6Mat = T.triu(M1s, 6) + T.tril(M1s, -6)
	LRsel = Sep24Mat.dimshuffle('x', 0, 1, 'x')
	MRsel = (Sep12Mat - Sep24Mat).dimshuffle('x', 0, 1, 'x')
	SRsel = (Sep6Mat - Sep12Mat).dimshuffle('x', 0, 1, 'x')

	selections = [LRsel, MRsel, SRsel]

	self.output = T.zeros((batchSize, seqLen, seqLen, n_out_max), dtype=theano.config.floatX)
	for emLayer, sel in zip(self.layers, selections):
	    l_n_out = emLayer.n_out
	    self.output = T.inc_subtensor(self.output[:, :, :, : l_n_out], T.mul(emLayer.output, sel) )

	self.pcenters = 0
        self.params = []
	self.paramL1 = 0
	self.paramL2 = 0	
	for layer in [ LRembedLayer, MRembedLayer, SRembedLayer]:
	    self.params += layer.params
	    self.paramL1 += layer.paramL1
	    self.paramL2 += layer.paramL2
	    self.pcenters += layer.pcenters

	self.n_out = n_out_max

class ProfileEmbeddingLayer(object):

    def __init__(self, input, n_in, n_out):
	##input has shape (batchSize, seqLen, n_in)
	##input is a profile derived from 1d convolution
	## n_in is the number of features in input
	
	W_value = np.float32(1.).astype(theano.config.floatX)
	self.W = theano.shared( value = W_value, name = 'ProfileEmbeddingLayer_scale', borrow=True)
	input2 = (input * self.W).dimshuffle(2, 0, 1).flatten(ndim=2).dimshuffle(1,0)
	input3 = T.nnet.softmax(input2).reshape(input.shape, ndim=3)
	embedLayer = MetaEmbeddingLayer(input3, n_in, n_out)

	self.output = embedLayer.output
	self.n_out = embedLayer.n_out
	self.params = [self.W ] + embedLayer.params
	self.paramL1 = abs(self.W) + embedLayer.paramL1
	self.paramL2 = (self.W**2) + embedLayer.paramL2
	self.pcenters = embedLayer.pcenters

        ##for test only, the input to T.nnet.softmax
        self.input_smax = input3

def TestProfileEmbeddingLayer():
    n_in = 3
    n_out = 5
    shape = (2, 10, n_in)
    a = np.random.uniform(-1, 1, shape)

    x = T.tensor3('x')
    layer = ProfileEmbeddingLayer( x, n_in, n_out)
    f = theano.function([x], [layer.output, layer.input_smax])
    b, smax = f(a)

    print a
    print smax
    print b

def TestEmbeddingLayer():

    n_in = 60
    a=np.random.uniform(0, 1, (20, 300, n_in)).round().astype(np.int32)
    n_out = 5

    x = T.itensor3('x')
    layer = MetaEmbeddingLayer(x, n_in, n_out)
    f = theano.function([x], [layer.output, layer.pcenters])

    b, pcenter = f(a)
   
    print b[0, 1, 2]
    print b[0, 1, 20]
    print a.shape
    batch=np.random.randint(0, 20)
    row1 = np.random.randint(0,100)
    row2 = np.random.randint(0,100)

    
    v1= a[batch][row1]
    v2= a[batch][row2]
    print b.shape
    print b[batch][row1][row2]
    c = np.outer( v1, v2)
    d = c[:, :, np.newaxis ]
    e = np.sum( (d * layer.W.get_value() ), axis=(0,1))
    print v1
    print v2
    print e
    print 'diff: ', abs(e - b[batch][row1][row2] ).sum()

    print pcenter
    center = [ np.sum( l.W.get_value(), axis=(0,1) ) for l in layer ] 
    print center
    print np.sum(center**2)

if __name__ == '__main__':

    #TestEmbeddingLayer()
    TestProfileEmbeddingLayer()
