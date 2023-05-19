from __future__ import print_function
import numpy as np
import sys
import time
from Util import *
from CLT_class import CLT
from numba import jit

class MIXTURE_CLT():
    
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = np.random.uniform(0, 1, 1) # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks  
      

    #Learn Mixtures of Trees using the EM algorithm.
    def learn(self, dataset, n_components=2, max_iter=50, epsilon=1e-5):
        num_samples = dataset.shape[0]
        self.n_components = n_components

        # For each component and each data point, we have a weight
        weights=np.zeros((n_components, dataset.shape[0]))
        # Randomly initialize the chow-liu trees and the mixture probabilities
        self.mixture_probs = np.random.uniform(0, 1, n_components)
        self.mixture_probs = self.mixture_probs / np.sum(self.mixture_probs)

        for itr in range(n_components):
            i_tree = CLT()
            i_tree.learn(dataset)
            init_w = np.random.uniform(0, 1, num_samples)
            init_w = init_w / np.sum(init_w)
            i_tree.update(dataset, init_w)
            self.clt_list.append(i_tree)

        for itr in range(max_iter):
            #E-step: Complete the dataset to yield a weighted dataset     
            gamma_i_k = np.zeros((n_components, num_samples))
            for i in range(num_samples):
                for k in range(n_components):
                    gamma_i_k[k, i] = self.clt_list[k].getProb(dataset[i])

            gamma_i_k *= self.mixture_probs[:, np.newaxis]
            gamma_i_k /= np.sum(gamma_i_k, axis=0)
            
            # M-step: Update the Chow-Liu Trees and the mixture probabilities   
            new_mix = np.zeros(n_components)
            for k in range(n_components):                
                new_mix[k] = np.sum(gamma_i_k[k,:]) / dataset.shape[0]
                self.clt_list[k].update(dataset, gamma_i_k[k,:])

            #print(self.mixture_probs)

            delta = np.sum(np.abs(new_mix - self.mixture_probs))
            self.mixture_probs = new_mix

            if delta < epsilon:
                break

    #Compute the log-likelihood score of the dataset
    def computeLL(self, dataset):
        ll = 0.0
        for i in range(dataset.shape[0]):
            p = 0
            for k in range(len(self.clt_list)):
                p = p + self.mixture_probs[k] * self.clt_list[k].getProb(dataset[i])
            ll = ll + np.log(p)
        return ll 
