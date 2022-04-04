import numpy as np


# class Normal:
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def pull(self):
#         return (np.random.randn() * self.std) + self.mean



class Bandits:
    def __init__(self, num_bandits, means=None, stdevs=None, walks=False):
        self.walks = walks
        if means is None:
            self.means = (np.random.rand(num_bandits)) + 2
        else:
            self.means = means

        if stdevs is None:
            self.stdevs = np.random.rand(num_bandits) * 0.3

    def pull(self, band):
        mean, stdev = self.means[band], self.stdevs[band]
        res= (np.random.randn() * stdev) + mean
        if self.walks:
            self.means[band] += np.random.randn()*0.01
        return res

