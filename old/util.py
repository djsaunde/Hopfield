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
        train_data[0][i] = np.array([ 1 if train_data[0][i][j] > 0 else -1 for j in range(len(train_data[0][1])) ])
        
    # preprocess testing image vectors to bipolar representation
    for i in range(len(test_data[0])):
        test_data[0][i] = np.array([ 1 if test_data[0][i][j] > 0 else -1 for j in range(len(test_data[0][1])) ])
    
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
    imshow(pattern.reshape((12, 12)), cmap = cm.binary, interpolation='nearest')
    
    # display the image
    show()
    
    
def bipolarize(pattern):
    '''
    change an image vector from binary format to bipolar format
    
    input:
        pattern: a (784,) binary array
        
    output:
        pattern, but in a bipolar format
    '''
    
    return np.array([ 1 if pattern[i] == 1 else -1 for i in range(len(pattern)) ])

    
def binarize(pattern):
    '''
    change an image vector from bipolar format to binary format.
    
    input:
        pattern: a (784,) bipolar array
        
    output:
        pattern, but in binary format
    '''
    
    return np.array([ 1 if pattern[i] == 1 else 0 for i in range(len(pattern)) ])
    
    
def sign_threshold(states):
    '''
    Takes a vector of state activations and thresholds it values to {-1, 1}.
    
    input:
        states: vector of dimension 'self.n_nodes' with values taken from [-1, 1]
        
    output:
        vector of dimension 'self.n_nodes' with values thresholded to -1 (if the
        component is < 0) or 1 (if the component is >= 1)
    '''
    
    # create empty states variable of the same shape as the previous states
    thresholded_states = np.zeros(states.shape)
    
    # loop through each component of the state vector and apply sign(x) function
    for i in range(len(states)):
        if states[i] >= 0:
            thresholded_states[i] = 1
        else:
            thresholded_states[i] = -1
         
    # return the thresholded activations   
    return thresholded_states
    

def add_noise(datum, p=0.15):
    '''
    Takes a vector of data and randomly flips its bipolar values with probability p.
    
    input:
        datum: a sample from the training dataset
        
    output:
        the sample corrupted with noise
    '''
    
    noisy_datum = np.zeros(datum.shape)
    
    # for each component of the data point...
    for i in range(len(datum)):
        # set the noisy datum equal to the original datum to begin
        noisy_datum[i] = datum[i]
        # if our random number is below the probability p...
        if np.random.rand() < p:
            # flip the bipolar value of this component
            noisy_datum[i] *= -1
    
    # return the noisy data point
    return noisy_datum
    
    
def build_dataset():
    '''
    build a toy alphabet dataset.
    '''
    
    a_pattern = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                          [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
 
    b_pattern = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
 
    c_pattern = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                          [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
                          
    d_pattern = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
                          
    e_pattern = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
                          
    f_pattern = np.array( [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                          
                          
    return np.array( [bipolarize(a_pattern.flatten()), bipolarize(b_pattern.flatten()), bipolarize(c_pattern.flatten()),
                      bipolarize(d_pattern.flatten()), bipolarize(e_pattern.flatten()), bipolarize(f_pattern.flatten())] )
    
    
    
