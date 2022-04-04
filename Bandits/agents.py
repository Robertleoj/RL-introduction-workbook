import numpy as np
import random
from typing import Union, Callable


class AgentBase:

    def get_action(self):
        raise NotImplementedError()

    def update(self, reward, action):
        raise NotImplementedError()


class EpsilonAgent:
    '''
    Agent that keeps a Q function, which it updates with
        Q_{n+1} = Q_n + lr(n)(R_n - Q_n)

    The user controls the learning rate lr(n)

    '''
    def __init__(self, 
            num_bandits: int, 
            epsilon: float, 
            lr: Union[float, Callable[[int], float]] =None
        ):
        '''
        Constructor.

        The learning rate is either a callable that receives n as input, or
        simply a floating point number for a constant learning rate.
        '''

        self.num_bandits = num_bandits
        self.qs = np.zeros(num_bandits) + 2
        self.ns = np.zeros(num_bandits)
        self.epsilon = epsilon

        # default is a mean agent
        if lr is None:
            self.lr = lambda n: 1/n

        # When the user simply passes in a floating point number
        elif not callable(lr):
            if not (isinstance(lr, int) or isinstance(lr, float)):
                raise ValueError("Learning rate is a function of n or a constant")
            self.lr = lambda n: lr
        else:
            self.lr = lr

    def get_action(self):
        """
        There is only one state, so just get an action
        """

        # epsilon greedy
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.num_bandits -1)

        return np.argmax(self.qs)

    def update(self, reward, action):
        """
        The agent needs to learn, so this must be called every move
        """
        self.ns[action] += 1
        self.qs[action] += self.lr(self.ns[action]) * (reward - self.qs[action])