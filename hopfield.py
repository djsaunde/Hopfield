'''
This script will handle the Hopfield network logic and associated functions.

author: Dan Saunders (djsaunde@umass.edu)
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

from util import *

class HopfieldNetwork(object):
    '''
    A Hopfield network object. 
    
    input: 
        num_nodes: the number of nodes / neurons in the network
        connectivity: a string which determines the interconnectedness of the 
            nodes in the network. this should be either 'full', 'lattice', or
            perhaps 'random', which is accompanied by a probability parameter 'p' 
    
    output:
        Newly instantiated Hopfield network object
    '''
    
    def __init__(self, n_nodes, connectivity='full', p=0.5):
        '''
        Initialize the Hopfield network.
        '''
        
        self.n_nodes = n_nodes
        self.weights = np.zeros((n_nodes, n_nodes))
        self.states = np.zeros((n_nodes))
       
        
    def train(self, train_images):
        '''
        Trains the Hopfield network via a matrix multiply of input vectors by their
        transposes.
        
        input:
            train_images: vectorized, preprocessed MNIST digit images
        '''
        
        # loop through each training sample
        for datum in train_images:
            # compute outer product and subtract identity matrix, add to weights matrix
            self.weights += np.outer(datum, datum.T)
            
        # diagonal of matrix should all be zero (no nodes connected to self)
        np.fill_diagonal(self.weights, 0)
            
        # divided the weight matrix by the number of training samples
        self.weights = np.divide(self.weights, len(train_images))
        
       
    def recall(self, patterns, steps=5):
        '''
        Iteratively tries to recall a pattern by reconstructing it from the weights 
        of the Hopfield network.
        
        input:
            patterns: some training samples we want to recall from the network
            steps: number of iterative steps to take in the recall process
            
        output:
            the recalled patterns from the network, to compare against the ground truth
        '''
        
        # for each step...
        for _ in range(steps):
            # iteratively update input patterns by dot product with weight matrix
            patterns = np.dot(patterns, self.weights)
            
        # return the patterns for inspection
        return patterns
        

        
if __name__ == '__main__':
    
    # get MNIST data
    train_data, test_data = get_MNIST()
    
    # unpack MNIST data into images, labels
    train_images, train_labels = train_data
    test_images, test_labels = test_data
    
    # cast these data into numpy arrays
    train_images, train_labels = np.array(train_images), np.array(train_labels)
    test_images, test_labels = np.array(train_images), np.array(train_labels)
    
    # get a subset of the training data so we don't run into a memory error at training time
    train_images, train_labels = train_images[0:500], train_labels[0:500]
    
    # get input dimension
    input_dim = train_images[0].shape[0]
    
    # create Hopfield network with 'n_nodes' = size of the input dimension
    hopnet = HopfieldNetwork(input_dim)
    
    # train hopfield network with training dataset
    hopnet.train(train_images)
    
    
    
    
    
    
