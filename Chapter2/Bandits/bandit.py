import numpy as np


class Bandits:
    def __init__(self, 
        num_bandits:int, 
        means_mean:float, 
        mean_std:float, 
        bandit_std:float,
        non_stationary:bool=False,
        rand_walk_std:float=None
    ):
        self.non_stationary = non_stationary
        self.rand_walk_std = rand_walk_std
        if non_stationary and rand_walk_std is None:
            raise ValueError(
                "If non_stationary = True then rand_walk_std must be specified"
            )
            
        self.means = (np.random.randn(num_bandits) * mean_std) + means_mean
        self.bandit_std = bandit_std

    def pull(self, band):
        mean = self.means[band]
        res= (np.random.randn() * self.bandit_std) + mean
        if self.non_stationary:
            self.means[band] += np.random.randn()*0.01
        return res
