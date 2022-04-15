from multiprocessing.sharedctypes import Value
from re import L
import numpy as np
import multiprocessing
from multiprocessing import Pool, Lock

from sympy import re
from utils import CompleteTaskCounter
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import ceil

NUM_THREADS = multiprocessing.cpu_count()
thread_counter = None


P_H = 0.55
P_T = 1 - P_H

GOAL = 100
THETA = 10e-4


def actions(state):
    for i in range(state + 1):
        yield i

def dynamics(state, action):
    """
    Generator function. 

    State is given as an integer, and action too, where 0 <= actionn <= state

    yields elements of the form
        newstate, reward, probability
    """
    # in terminal state
    if state == 0:
        yield 0, 0, 1
        return

    if state == GOAL:
        yield 0, 0, 1
        return

    # lose
    yield state - action, 0, P_T
    # wiin
    winstate = state + action
    if winstate >= GOAL:
        yield GOAL, 1, P_H
    else:
        yield winstate, 0, P_H

def value_iter_thread(args):
    """
    Performs one iteration of the value iteration loop

    returns the new value, delta, and the state

    we need the state to do the update, as the thing is threaded
    """
    c, values, state = args
    v_orig = values[state]
    best = -1000000
    for a in actions(state):
        val = 0
        for newstate, reward, p in dynamics(state, a):
            val += p * (reward + values[newstate])
        best = max(val, best)

    # global thread_counter
    # thread_counter.incrprint(c)

    return best, abs(v_orig - best), state

def value_iteration(values):
    global thread_counter
    thread_counter = CompleteTaskCounter(GOAL + 1)
    def args():
        for i in range(GOAL + 1):
            yield (thread_counter.count, values, i)

    res = None
    with Pool(NUM_THREADS) as pool:
        res = pool.map(value_iter_thread, args())

    delta = 0
    for best, d, state in res:
        delta = max(d, delta)
        values[state] = best
    print(delta)
    return delta < THETA
    

def create_policy(values):
    policy = np.zeros(GOAL+ 1, int)

    for state in range(GOAL + 1):
        best = -1000000
        best_action = None
        for a in actions(state):
            val = 0
            for newstate, reward, p in dynamics(state, a):
                val += p * (reward + values[newstate])
            if val > best:
                best = val
                best_action = a
        policy[state] = best_action
    return policy


def plot_policy(policy):
    fig, ax = plt.subplots()
    ax.plot(policy)
    ax.set_ylim((0, GOAL))
    ax.set_ylabel("Stake")
    ax.set_xlabel("Capital")
    fig.savefig(f'Ph{P_H}_policy.png')


class ValuePlotter:
    def __init__(self):
        self.iters = 0
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("State-Value function")
        self.prev_plot = None


    def should_plot(self):
        if self.prev_plot is None:
            return True
        return self.iters >= ceil(self.prev_plot*1.4)

    
    def add_line(self, values):
        self.iters += 1
        if self.should_plot():
            self.ax.plot(values[1:GOAL], label=f"sweep {self.iters}")
            self.prev_plot = self.iters

    def end(self):
        self.ax.legend()
        self.ax.set_ylim((0, 1))
        self.fig.savefig(f'Ph{P_H}_values.png')

def main():
    values = np.zeros(GOAL + 1)

    vplotter = ValuePlotter()

    stable = False
    iters = 0
    while not stable:
        iters += 1
        stable = value_iteration(values)
        vplotter.add_line(values)
        print(f"{iters = }")

    vplotter.add_line(values)
    vplotter.end()
    
    policy = create_policy(values)

    plot_policy(policy)

main()



