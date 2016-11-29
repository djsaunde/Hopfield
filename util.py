'''
Helper methods for the 'hopfield.py' script.

author: Dan Saunders (djsaunde@umass.edu)
'''

import numpy as np

from mnist import MNIST
from pylab import imshow, cm, show


def get_MNIST():
    '''
    returns training, testing MNIST data.
    
    output: (train_data, test_data) tuple
    '''
    
    # create an MNIST object, giving directory where the MNIST data is stored
    mndata = MNIST('/home/dan/Code/data')
    
    # get training, testing data
    train_data, test_data = mndata.load_training(), mndata.load_testing()
    
    # preprocess training image vectors to bipolar representation
    for i in range(len(train_data[0])):
        train_data[0][i] = [ 1 if train_data[0][i][j] > 0 else -1 for j in range(len(train_data[0][i])) ]
        
    # preprocess testing image vectors to bipolar representation
    for i in range(len(test_data[0])):
        test_data[0][i] = [ 1 if test_data[0][i][j] > 0 else -1 for j in range(len(test_data[0][i])) ]
    
    # return the datasets
    return train_data, test_data
    
    
def display(pattern):
    '''
    display a binary image given by pattern, of shape (784,), which is reshaped to
    (28, 28).
    
    input:
        pattern: a (784,) binary array
        
    output:
        image displayed using scipy imshow function
    '''
    
    # use imshow to display to user
    imshow(pattern.reshape((28, 28)), cmap = cm.binary, interpolation='nearest')
    show()
    
    
def binarize(pattern):
    '''
    change a image vector from bipolar format to binary format.
    
    input:
        pattern: a (784,) bipolar array
        
    output:
        pattern, but in binary format
    '''
    
    return [ 1 if pattern[i] == 1 else -1 for i in range(len(pattern)) ]
    
    
def sign_threshold(self, states):
    '''
    Takes a vector of state activations and thresholds it values to {-1, 1}.
    
    input:
        states: vector of dimension 'self.n_nodes' with values taken from [-1, 1]
        
    output:
        vector of dimension 'self.n_nodes' with values thresholded to -1 (if the
        component is < 0) or 1 (if the component is >= 1)
    '''
    
    # loop through each component of the state vector and apply sign(x) function
    for i in range(len(states)):
        if states[i] >= 0:
            states[i] = 1
        else:
            states[i] = -1
         
    # return the thresholded activations   
    return states
    
    
    
