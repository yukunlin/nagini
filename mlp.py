""" MLP in Python """
__author__ = 'Teo Asinari'
__date__ = '2014-02-19'

import numpy
import math
from numpy import *
from numpy.random import *
from numpy.matrixlib import *


# Activation functions
def tansig(x):
    return math.tanh(x)


def logsig(x):
    return 1/(1+math.exp(-float(x)))


def hardlim(x):
    return 0.5 * (sign(x) + 1)


def hardlims(x):
    return sign(x)

# activtion function and derivative
def afun(name, x, discrete_flag):
    if discrete_flag == 0:
        return {
        'tansig': lambda y: tansig(y),
        'logsig': lambda y: logsig(y),
        'purelin': lambda y: y,
        'hardlim': lambda y: hardlim(y),
        'hardlims': lambda y: hardlims(y)
        }[name](x)
    else:
        return {
        'tansig': lambda y: sign(tansig(y)),
        'logsig': lambda y: hardlim(x-0.5),
        'purelin': lambda y: sign(y),
        'hardlim': lambda y: hardlim(y),
        'hardlims': lambda y: hardlims(y)
        }[name](x)


def adfun(name, x):
    return {
        'tansig': lambda y: 1-math.pow(tansig(y), 2),
        'logsig': lambda y: logsig(y)*(1-logsig(y)),
        'purelin': lambda y: 1,
        'hardlim': 'error no deriv',
        'hardlims': 'error no deriv'
    }[name](x)

# to make numpy happy ...
afun_vec = vectorize(afun)
adfun_vec = vectorize(adfun)

class MLP:
    def __init__(self, input_size=0, neuron_count=0, activfuncts=[]):
        self.input = []
        self.input_size = input_size
        self.desired = []

        self.discrete_out_flag = 0

        self.num_iters = 0
        self.iters_lim = 0

        self.curr_sample_ind = 0
        self.curr_iter = 0

        self.neuron_count = neuron_count
        self.num_layers = len(neuron_count)

        dummy = range(self.num_layers)

        self.learning_rate = 0
        self.weights = []
        self.netvals = list(dummy)
        self.activvals = list(dummy)
        self.activfuncts = activfuncts
        self.bias = []

# Generate weights and bias for the network.
    def genWB(self, valrange):

        for i in range(self.num_layers):

            if i == 0:
                inputs_len = self.input_size
            else:
                inputs_len = self.neuron_count[i-1]

            W_random_matrix = asmatrix(random_sample((self.neuron_count[i], inputs_len)))

            b_random_matrix = asmatrix(random_sample((self.neuron_count[i], 1)))

            self.weights.append(valrange * (W_random_matrix - valrange / 2))

            self.bias.append(valrange * (b_random_matrix - valrange / 2))

# Updaters
    def nup(self, curr_layer):
        input_layer = (curr_layer == 0)

        if input_layer:
            layer_input = self.input
        else:
            layer_input = self.activvals[curr_layer-1]

        self.netvals[curr_layer] = ((self.weights[curr_layer] * layer_input) + self.bias[curr_layer])

    def aup(self, curr_layer):
        self.nup(curr_layer)

        curr_afun = self.activfuncts[curr_layer]

        self.activvals[curr_layer] = (afun_vec(curr_afun,
                                self.netvals[curr_layer],
                                self.discrete_out_flag))

    def aups(self, input):
        self.input = input
        for layer in xrange(self.num_layers):
            self.aup(layer)

    def mlp_output(self):
        return self.activvals[self.num_layers-1]

def main():
    a = MLP(input_size=3, neuron_count=[3, 2], activfuncts=['tansig', 'tansig'])
    a.genWB(0.5)
    a.aups(array([[1], [2], [3]]))
    print a.mlp_output()

if __name__ == "__main__":
    main()