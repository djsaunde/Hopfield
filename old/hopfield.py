'''
This script will handle the Hopfield network logic and associated functions.

author: Dan Saunders (djsaunde@umass.edu)
'''

import numpy as np
import matplotlib.pyplot as plt
import random

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
    
    def __init__(self, n_nodes, connectivity='full'):
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
            self.weights += np.outer(datum, datum)
            
        # diagonal of matrix should all be zero (no nodes connected to self)
        np.fill_diagonal(self.weights, 0)
            
        # divided the weight matrix by the number of training samples
        self.weights = np.divide(self.weights, len(train_images))
        
        print self.weights
        
       
    def recall(self, pattern, steps=5):
        '''
        Iteratively tries to recall a pattern by reconstructing it from the weights 
        of the Hopfield network.
        
        input:
            pattern: some noisy training sample we want to recall from the network
            steps: number of iterative steps to take in the recall process
            
        output:
            the recalled patterns from the network, to compare against the ground truth
        '''
        
        # for each step...
        for _ in range(steps):
            # iteratively update input pattern by dot product with weight matrix
            pattern = sign_threshold(np.dot(pattern, self.weights))
            
        # return the pattern for inspection
        return pattern
        
        
    def energy(self, pattern):
        '''
        Returns the "energy" of a certain pattern according to the network.
        
        input:
            pattern: vector whose energy we wish to discover
            
        output:
            the energy of the input pattern
        '''
        
        # do the energy calculation and return
        return -0.5 * np.dot(np.dot(pattern.T, self.weights), pattern) 

        
if __name__ == '__main__':
    
    '''
    # get MNIST data
    train_data, test_data = get_MNIST()
    
    # unpack MNIST data into images, labels
    train_images, train_labels = train_data
    test_images, test_labels = test_data
    
    # cast these data into numpy arrays
    train_images, train_labels = np.array(train_images), np.array(train_labels)
    test_images, test_labels = np.array(train_images), np.array(train_labels)
    
    # get a subset of the training data so we don't run into a memory error at training time
    train_images, train_labels = train_images[0:2], train_labels[0:2]
    '''
    
    # get training images
    train_images = build_dataset()
    
    # get input dimension
    input_dim = train_images[0].shape[0]
    
    # create Hopfield network with 'input_dim' = size of the input dimension
    hopnet = HopfieldNetwork(input_dim)
    
    # train hopfield network with training dataset
    hopnet.train(train_images)
    
    # recall some noisy images
    for _ in range(5):
        # pick a random data point
        datum = random.choice(train_images)
        # display it
        display(binarize(datum))
        # add some noise to the sample
        noisy_datum = add_noise(datum)
        # display it
        display(binarize(noisy_datum))
        # run the recall function
        recalled_datum = hopnet.recall(noisy_datum)
        # display it
        display(recalled_datum)
    
    # get energies of training patterns    
    for img, letter in zip(train_images, ['A', 'B', 'C', 'D', 'E', 'F']):
        print hopnet.energy(img), letter
        
    
    
    
    
    
    
