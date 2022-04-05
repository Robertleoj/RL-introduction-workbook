import numpy as np
import random
from typing import Union, Callable
from math import log, sqrt


class AgentBase:

    def get_action(self):
        raise NotImplementedError()

    def update(self, reward, action):
        raise NotImplementedError()

    def _init_lr(self, lr):
        if lr is None:
            self.lr = lambda n: 1/n

        # When the user simply passes in a floating point number
        elif not callable(lr):
            if not (isinstance(lr, int) or isinstance(lr, float)):
                raise ValueError("Learning rate is a function of n or a constant")
            self.lr = lambda n: lr
        else:
            self.lr = lr




class EpsilonAgent(AgentBase):
    '''
    Agent that keeps a Q function, which it updates with
        Q_{n+1} = Q_n + lr(n)(R_n - Q_n)

    The user controls the learning rate lr(n)

    This agent uses epsilon greedy
    '''
    def __init__(self, 
            num_bandits: int, 
            epsilon: float, 
            initial_q: float,
            lr: Union[float, Callable[[int], float]] =None
        ):
        '''
        Constructor.

        The learning rate is either a callable that receives n as input, or
        simply a floating point number for a constant learning rate.
        '''
        super().__init__()
        self.num_bandits = num_bandits
        self.qs = np.zeros(num_bandits) + initial_q
        self.ns = np.zeros(num_bandits)
        self.epsilon = epsilon

        # default is a mean agent
        self._init_lr(lr)

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



class UCBAgent(AgentBase):
    '''
    Agent that keeps a Q function, which it updates with
        Q_{n+1} = Q_n + lr(n)(R_n - Q_n)

    The user controls the learning rate lr(n)

    This agent uses the UCB selection method
    '''
    def __init__(self, 
            num_bandits: int, 
            initial_q: float,
            c: float,
            lr: Union[float, Callable[[int], float]] =None,
        ):
        '''
        Constructor.

        The learning rate is either a callable that receives n as input, or
        simply a floating point number for a constant learning rate.

        c is the hyperparameter for the UCB function
        '''

        super().__init__()

        self.curr_step = 0
        self.num_bandits = num_bandits
        self.qs = np.zeros(num_bandits) + initial_q
        self.ns = np.zeros(num_bandits)

        if c is None:
            raise ValueError("constant c cannot be none")

        self.UCBc = c

        self._init_lr(lr)

    def _UCB(self, action):
        return self.qs[action] + self.UCBc * sqrt(log(self.curr_step) / self.ns[action])

    def get_action(self):
        """
        There is only one state, so just get an action
        """ 

        max_ucb = -10000
        best_action = None
        for action in range(self.num_bandits):
            if self.ns[action] == 0:
                return action # untried actions considered optimal

            ucb = self._UCB(action)
            if ucb > max_ucb:
                max_ucb = ucb
                best_action = action

        return best_action

    def update(self, reward, action):
        """
        The agent needs to learn, so this must be called every move
        """
        self.curr_step += 1
        self.ns[action] += 1
        self.qs[action] += self.lr(self.ns[action]) * (reward - self.qs[action])

class GradientAgent(AgentBase):
    '''
    Agent that keeps a Q function, which it updates with
        Q_{n+1} = Q_n + lr(n)(R_n - Q_n)

    The user controls the learning rate lr(n)

    This agent uses the UCB selection method
    '''
    def __init__(self, 
            num_bandits: int, 
            lr: Union[float, Callable[[int], float]] =None,
        ):
        '''
        Constructor.

        The learning rate is either a callable that receives n as input, or
        simply a floating point number for a constant learning rate.

        c is the hyperparameter for the UCB function
        '''

        super().__init__()

        self.avgreward = 0
        self.curr_step = 0

        self.num_bandits = num_bandits
        self.Hs = np.zeros(num_bandits)
        self._compute_policy()

        self._init_lr(lr)

    def _compute_policy(self):
        exps = np.exp(self.Hs) 
        self.policy = exps / exps.sum()

    def get_action(self):
        """
        There is only one state, so just get an action
        """ 
        return np.random.choice(range(self.num_bandits), 1, p=self.policy)[0]


    def update(self, reward, action):
        """
        The agent needs to learn, so this must be called every move
        """
        self.curr_step += 1

        # Update the avg reward (baseline)
        n = self.curr_step
        self.avgreward = (n / (n + 1)) * self.avgreward + (1 / (n + 1))* reward

        # perform updates
        step_mult = self.lr(n) * (reward - self.avgreward)
        for i in range(self.num_bandits):
            if i == action:
                self.Hs[i]+= step_mult * (1 - self.policy[i])
            else:
                self.Hs[i] -= step_mult * self.policy[i]

        self._compute_policy()








