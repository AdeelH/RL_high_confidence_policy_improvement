import numpy as np
from .bbo_agent import BBOAgent
from .utils import top_k_inds

from typing import Callable
from multiprocessing import Pool
from functools import partial
from itertools import product

class CEM(BBOAgent):
    """
    The cross-entropy method (CEM) for policy search is a black box optimization (BBO)
    algorithm. This implementation is based on Stulp and Sigaud (2012). Intuitively,
    CEM starts with a multivariate Gaussian dsitribution over policy parameter vectors.
    This distribution has mean thet and covariance matrix Sigma. It then samples some
    fixed number, K, of policy parameter vectors from this distribution. It evaluates
    these K sampled policies by running each one for N episodes and averaging the
    resulting returns. It then picks the K_e best performing policy parameter
    vectors and fits a multivariate Gaussian to these parameter vectors. The mean and
    covariance matrix for this fit are stored in theta and Sigma and this process
    is repeated.

    Parameters
    ----------
    sigma (float): exploration parameter
    theta (numpy.ndarray): initial mean policy parameter vector
    popSize (int): the population size
    numElite (int): the number of elite policies
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates the provided parameterized policy.
        input: theta_p (numpy.ndarray, a parameterized policy), numEpisodes
        output: the estimated return of the policy
    epsilon (float): small numerical stability parameter
    """

    def __init__(self, theta:np.ndarray, sigma:float, popSize:int, numElite:int, evaluationFunction:Callable, epsilon:float=0.0001):

        self._theta = theta
        self._Sigma = np.eye(len(theta)) * sigma
        self.popSize = popSize
        self.numElite = numElite
        self.evaluationFunction = evaluationFunction
        self.epsilon = epsilon
        
        self._theta_initial = self._theta.copy()
        self._Sigma_initial = self._Sigma.copy()

    @property
    def name(self)->str:
        return 'CEM'
    
    @property
    def parameters(self)->np.ndarray:
        return self._theta.copy()

    def train(self)->np.ndarray:
        K, K_e = self.popSize, self.numElite
        eps, dim = self.epsilon, self._theta.shape[0]

        thetas = np.random.multivariate_normal(self._theta, self._Sigma, size=K)
        # if log_evals:
        #     tmp = [self.evaluationFunction(theta, log_evals=log_evals) for theta in thetas]
        #     evals, log = zip(*tmp)
        # else:
        #     evals = [self.evaluationFunction(theta) for theta in thetas]

        evals = [self.evaluationFunction(theta) for theta in thetas]
        print(evals)
        top_thetas = thetas[top_k_inds(evals, K_e)]
        self._theta = top_thetas.mean(axis=0)
        centered = top_thetas - self._theta
        self._Sigma = (1/(eps + K_e)) * (np.eye(dim) * eps + np.einsum('ij,ik->jk', centered, centered))

        return self._theta

    def reset(self)->None:
        self._theta = self._theta_initial.copy()
        self._Sigma = self._Sigma_initial.copy()
