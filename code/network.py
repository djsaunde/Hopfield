"""
This file implements a Hopfield network. It provides functions to
set and retrieve the network state, store patterns.

Relevant book chapters:
    - http://neuronaldynamics.epfl.ch/online/Ch17.S2.html

"""

# re-purposing the Hopfield network code from the Neuronal Dynamics textbook exercises.

import numpy as np
import random
import math


class HopfieldNetwork:
    """
    Implements a Hopfield network.

    Attributes:
        nrOfNeurons (int): Number of neurons
        weights (numpy.ndarray): nrOfNeurons x nrOfNeurons matrix of weights
        state (numpy.ndarray): current network state. matrix of shape (nrOfNeurons, nrOfNeurons)
    """

    def __init__(self, nr_neurons, connectivity='full'):
        """
        Constructor for the Hopfield network.

        Args:
            nr_neurons (int): Number of neurons. Use a square number to get the
            visualizations properly.
        """
        # math.sqrt(nr_neurons)
        self.nrOfNeurons = nr_neurons
        # connectivity pattern
        self.connectivity = connectivity
        # initialize with random state
        self.state = 2 * np.random.randint(0, 2, self.nrOfNeurons) - 1
        # initialize random weights
        self.reset_weights()
        # set update function to the synchronous and deterministic sign(h) function
        self.set_dynamics_sign_sync()
        

    def reset_weights(self):
        """
        Resets the weights to random values.
        """
        self.weights = 1.0 / self.nrOfNeurons * (2 * np.random.rand(self.nrOfNeurons, self.nrOfNeurons) - 1)
            

    def set_dynamics_sign_sync(self):
        """
        Sets the update dynamics to the synchronous, deterministic g(h) = sign(h) function.
        """
        self._update_method = _get_sign_update_function()
        

    def set_dynamics_sign_async(self):
        """
        Sets the update dynamics to the g(h) = sign(h) functions. Neurons are updated asynchronously:
        in random order, all neurons are updated sequentially.
        """
        self._update_method = _get_async_sign_update_function()
        
    
    def set_dynamics_tanh_sync(self):
        """
        Sets the update dynamics to the g(h) = 0.5 * [ 1 + tanh(beta * h) ] functions. Neurons are
        update synchronously.
        """
        self._update_method = _get_tanh_update_function()
        
    
    def set_dynamics_tanh_async(self):
        """
        Sets the update dynamics to the g(h) = 0.5 * [ 1 + tanh(beta * h) ] functions. Neurons are
        update asynchronously: in random order, all neurons are updated sequentially.
        """
        self._update_method = _get_async_tanh_update_function()
        

    def set_dynamics_to_user_function(self, update_function):
        """
        Sets the network dynamics to the given update function

        Args:
            update_function: upd(state_t0, weights) -> state_t1.
                Any function mapping a state s0 to the next state
                s1 using a function of s0 and weights.
        """
        self._update_method = update_function
        

    def store_patterns(self, pattern_list):
        """
        Learns the patterns by setting the network weights. The patterns
        themselves are not stored, only the weights are updated!
        self connections are set to 0.

        Args:
            pattern_list: a nonempty list of patterns.
        """
        all_same_size_as_net = all(len(p.flatten()) == self.nrOfNeurons for p in pattern_list)
        if not all_same_size_as_net:
            errMsg = "Not all patterns in pattern_list have exactly the same number of states " \
                     "as this network has neurons n = {0}.".format(self.nrOfNeurons)
            raise ValueError(errMsg)
        
        self.weights = np.zeros((self.nrOfNeurons, self.nrOfNeurons))
        # textbook formula to compute the weights:
        for p in pattern_list:
            p_flat = p.flatten()
            for i in range(self.nrOfNeurons):
                for k in range(self.nrOfNeurons):
                    self.weights[i, k] += p_flat[i] * p_flat[k]
        
        if self.connectivity == 'lattice':            
            sqrt = math.sqrt(self.nrOfNeurons)
            for i in range(self.nrOfNeurons):
                for k in range(self.nrOfNeurons):
                    if not (i + 1 == k and k % sqrt != 0 or i - 1 == k and i % sqrt != 0 or i + sqrt == k or i - sqrt == k):
                        self.weights[i, k] = 0
                
        self.weights /= self.nrOfNeurons
        # no self connections:
        np.fill_diagonal(self.weights, 0)
        

    def set_state_from_pattern(self, pattern):
        """
        Sets the neuron states to the pattern pixel. The pattern is flattened.

        Args:
            pattern: pattern
        """
        self.state = pattern.copy().flatten()
        

    def iterate(self):
        """
        Executes one timestep of the dynamics.
        """
        self.state = self._update_method(self.state, self.weights)
        

    def run(self, nr_steps=5):
        """
        Runs the dynamics.

        Args:
            nr_steps (float, optional): Timesteps to simulate
        """
        for i in range(nr_steps):
            # run a step
            self.iterate()
            

    def run_with_monitoring(self, nr_steps=5):
        """
        Iterates for nr_steps steps. Records the network state after every
        iteration.

        Args:
            nr_steps (float, optional): Timesteps to simulate

        Returns:
            A list of 2D network states
        """
        states = list()
        states.append(self.state.copy())
        for i in range(nr_steps):
            # run a step
            self.iterate()
            states.append(self.state.copy())
        return states


def _get_sign_update_function():
    """
    Gives the synchronous sign update function.

    Returns:
        A function implementing a synchronous state update using sign(h)
    """
    def upd(state_s0, weights):
        h = np.sum(weights * state_s0, axis=1)
        s1 = np.sign(h)
        # by definition, neurons have state +/-1. If the sign function returns 0, we set it to +1
        idx0 = s1 == 0
        s1[idx0] = 1
        return s1
    return upd


def _get_async_sign_update_function():
    """
    Gives the asynchronous sign update function.
    
    Returns:
        A function implementing an asynchronous state update using sign(h)
    """
    def upd(state_s0, weights):
        random_neuron_idx_list = np.random.permutation(len(state_s0))
        state_s1 = state_s0.copy()
        for i in range(len(random_neuron_idx_list)):
            rand_neuron_i = random_neuron_idx_list[i]
            h_i = np.dot(weights[:, rand_neuron_i], state_s1)
            s_i = np.sign(h_i)
            if s_i == 0:
                s_i = 1
            state_s1[rand_neuron_i] = s_i
        return state_s1
    return upd
    
def _get_tanh_update_function(beta):
    """
    Gives the synchronous tanh update function.
    
    Returns:
        A function implementing the synchronous state update using g(h) = 0.5[1 + tanh(beta*h)].
    """
    def upd(state_s0, weights):
        h = np.sum(weights * state_s0, axis=1)
        s1 = 0.5 * (1 + np.tanh(beta * h))
        # flip a random coin; if it's less than s1 (prob that state becomes 1), set to 1; otherwise, set to -1
        if random.random() <= s1:
            s1 = 1
        else:
            s1 = -1
        return s1
    return upd
    
def _get_async_anh_update_function(beta):
    """
    Gives the asynchronous tanh update function.
    
    Returns:
        A function implementing the asynchronous state update using g(h) = 0.5[1 + tanh(beta*h)].
    """
    def upd(state_s0, weights):
        random_neuron_idx_list = np.random.permutation(len(state_s0))
        state_s1 = state_s0.copy()
        for i in range(len(random_neuron_idx_list)):
            rand_neuron_i = random_neuron_idx_list[i]
            h_i = np.dot(weights[:, rand_neuron_i], state_s1)
            s_i = np.tanh(h_i)
            if random.random <= s_i:
                s_i = 1
            else:
                s_i = -1
            state_s1[rand_neuron_i] = s_i
        return state_s1
    
        h = np.sum(weights * state_s0, axis=1)
        s1 = 0.5 * (1 + np.tanh(beta * h))
        # flip a random coin; if it's less than s1 (prob that state becomes 1), set to 1; otherwise, set to -1
        if random.random() <= s1:
            s1 = 1
        else:
            s1 = -1
        return s1
    return upd

