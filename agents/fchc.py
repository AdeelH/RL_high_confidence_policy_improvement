import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class FCHC(BBOAgent):
    """
    First-choice hill-climbing (FCHC) for policy search is a black box optimization (BBO)
    algorithm. This implementation is a variant of Russell et al., 2003. It has 
    remarkably fewer hyperparameters than CEM, which makes it easier to apply. 
    
    Parameters
    ----------
    sigma (float): exploration parameter 
    theta (np.ndarray): initial mean policy parameter vector
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates a provided policy.
        input: policy (np.ndarray: a parameterized policy), numEpisodes
        output: the estimated return of the policy 
    """

    def __init__(self, theta:np.ndarray, sigma:float, evaluationFunction:Callable):
        
        self._theta = theta
        self._Sigma = np.eye(len(theta)) * sigma
        self.evaluationFunction = evaluationFunction

        self._theta_initial = self._theta.copy()
        self.eval = self.evaluationFunction(self._theta)

    @property
    def name(self)->str:
        return 'fchc'
    
    @property
    def parameters(self)->np.ndarray:
        return self._theta.copy()

    def train(self)->np.ndarray:
        theta_new = np.random.multivariate_normal(self._theta, self._Sigma)

        eval_new = self.evaluationFunction(theta_new)

        if eval_new > self.eval:
            self._theta = theta_new
            self.eval = eval_new

        return self._theta


    def reset(self)->None:
        self._theta = self._theta_initial.copy()
        self.eval = self.evaluationFunction(self._theta, self.numEpisodes)
