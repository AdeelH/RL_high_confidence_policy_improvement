import numpy as np
from .bbo_agent import BBOAgent
from .utils import top_k_inds

from typing import Callable


class GA(BBOAgent):
    """
    A canonical Genetic Algorithm (GA) for policy search is a black box 
    optimization (BBO) algorithm. 
    
    Parameters
    ----------
    populationSize (int): the number of individuals per generation
    numEpisodes (int): the number of episodes to sample per policy         
    evaluationFunction (function): evaluates a parameterized policy
        input: a parameterized policy theta, numEpisodes
        output: the estimated return of the policy            
    initPopulationFunction (function): creates the first generation of
                    individuals to be used for the GA
        input: populationSize (int)
        output: a numpy matrix of size (N x M) where N is the number of 
                individuals in the population and M is the number of 
                parameters (size of the parameter vector)
    numElite (int): the number of top individuals from the current generation
                    to be copied (unmodified) to the next generation
    
    """

    def __init__(self, populationSize:int, evaluationFunction:Callable, 
                 initPopulationFunction:Callable, numElite:int=1, K_p=None, alpha=1.):

        self.populationSize = populationSize
        self.evaluationFunction = evaluationFunction
        self.initPopulationFunction = initPopulationFunction
        self.K_e = numElite
        self.K_p = K_p if K_p is not None else populationSize
        self.alpha = alpha
        self.numGeneratedChildren = populationSize - numElite

        self.reset()
        self.dim = self._population[0].shape[0]

    @property
    def name(self)->str:
        return 'GA'
    
    @property
    def parameters(self)->np.ndarray:
        return self._theta.copy()

    def _mutate(self, parent:np.ndarray)->np.ndarray:
        """
        Perform a mutation operation to create a child for the next generation.
        The parent must remain unmodified. 
        
        output:
            child -- a mutated copy of the parent
        """
        noise = np.random.randn(parent.shape[0])
        child = parent + self.alpha * noise
        return child

    def _get_children(self, parents):
        n = self.numGeneratedChildren
        sampled_parents = parents[np.random.choice(len(parents), size=n)]
        noise = self.alpha * np.random.randn(n, self.dim)
        children = sampled_parents + self.alpha * noise
        return children

    def train(self)->np.ndarray:
        K_e, K_p = self.K_e, self.K_p
        
        parents = self._population[top_k_inds(self._evals, K_p)]
        next_gen = self._get_children(parents)
        if K_e > 0:
            elites  = self._population[top_k_inds(self._evals, K_e)]
            self._population = np.concatenate((elites, next_gen), axis=0)
        else:
            self._population = next_gen

        evals = [self.evaluationFunction(theta) for theta in self._population]

        self._evals = evals
        self._theta = self._population[np.argmax(self._evals)]

        return self._theta

    def reset(self)->None:
        self._population = self.initPopulationFunction(self.populationSize)
        self._evals = [self.evaluationFunction(theta) for theta in self._population]
        self._theta = self._population[np.argmax(self._evals)]
