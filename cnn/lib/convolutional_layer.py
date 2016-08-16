import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class LeNetConvPoolLayer(object):
    def single_conv(self, input):
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape
        )
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=self.poolsize,
            ignore_border=True
        )
        return self.activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
       
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), activation=T.tanh):
        assert image_shape[1] == filter_shape[1]
        
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation = activation
        
        #W
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        #b
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        
        #conv output
        self.output = tuple(self.single_conv(i) for i in input)

        # store parameters of this layer
        self.params = [self.W, self.b]


