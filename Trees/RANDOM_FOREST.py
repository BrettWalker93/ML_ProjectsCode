import numpy as np
from Util import *
from CLT_class import CLT

class RANDOM_FOREST():
  def __init__ (self):
    self.n_components = 0 # number of components
    self.clt_list = []
    self.r = 0 #number of edgemats to set to 0

  def learn(self, dataset, n_components = 2, r = 5):
    self.n_components = n_components

    for itr in range(n_components):
      i_tree = CLT()
      bootset = dataset.copy()
      for i in range(dataset.shape[0]):
        bootset[i] = dataset[np.random.randint(0, dataset.shape[0])]
      r_i = np.random.randint(0, dataset.shape[1], r)
      r_j = np.random.randint(0, dataset.shape[1], r)
      i_tree.learn_r(dataset, r_i, r_j)
      self.clt_list.append(i_tree)

  def computeLL(self, dataset):
      ll = 0.0
      for i in range(dataset.shape[0]):
          p = 0
          for k in range(self.n_components):
              p = p + self.clt_list[k].getProb(dataset[i])
          p = p / self.n_components
          ll = ll + np.log(p)
      return ll 
