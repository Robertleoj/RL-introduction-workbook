# Cell
from bandit import Bandits
import numpy as np
import random
from threading import Lock
import concurrent.futures
import matplotlib.pyplot as plt
import os


AVAIL_THREADS =  (int)(os.popen('grep -c cores /proc/cpuinfo').read())

NUM_THREADS_LEFT = 0
LR = 0.01
# LR = None
USE_LINSPACE_MEANS = False

mutex = Lock()


class Agent:
    def __init__(self, num_bandits, epsilon, lr=None):
        self.num_bandits = num_bandits
        self.qs = np.zeros(num_bandits)
        self.ns = np.zeros(num_bandits)
        self.epsilon = epsilon
        if lr is None:
            self.lr = lambda n: 1/n
        else:
            self.lr = lambda n: lr

    def get_action(self):

        if np.random.rand() < self.epsilon:
            return random.randint(0, self.num_bandits -1)

        return np.argmax(self.qs)

    def update(self, reward, action):
        self.ns[action] += 1
        # print(self.lr(self.ns[action]))
        self.qs[action] += self.lr(self.ns[action]) * (reward - self.qs[action])

def smooth(cumscores, num_samples):
    new = cumscores.copy()
    for score_ro in new:
        for i in range(len(new)-1, num_samples, -1 ):
            slice = score_ro[i - num_samples: i]
            score_ro[i] = slice.mean()

    return new


def experiment(num_bandits= 10, num_agents=5, epsilons=None, num_actions=10000, non_stationary=False):
    means =  np.linspace(0, 2, num_bandits) if USE_LINSPACE_MEANS else None
    # print(means)
    bandit = Bandits(num_bandits, means, None, non_stationary)
    if epsilons is None:
        epsilons = np.linspace(0, 0.15, num_agents)

    agents = list(map(lambda x: Agent(num_bandits, x,LR), epsilons))

    scores = np.zeros((num_agents, num_actions))

    for n in range(num_actions):
        for i,agent in enumerate(agents):
            move = agent.get_action()
            reward = bandit.pull(move)
            agent.update(reward, move)
            scores[i,n] = reward

    return scores

def experiment_thread(args):
    (   
        i,scores, num_bandits, num_agents, epsilons, num_actions, non_stationary
    ) = args
    # print(f"Thread {i} starting...")
    s = experiment(num_bandits, num_agents, epsilons, num_actions, non_stationary)
    scores[i,:,:] = s
    # print(f"Thread {i} completed!")
    global NUM_THREADS_LEFT
    mutex.acquire()
    NUM_THREADS_LEFT -= 1
    print(f"Num threads left = {NUM_THREADS_LEFT}")
    mutex.release()


def main():
    num_agents = 5
    num_bandits = 10
    epsilons = [0,0.001, 0.01, 0.05, 0.1]
    num_actions = 1000
    num_experiments = 1000
    non_stationary = True

    scores = np.zeros((num_experiments, num_agents, num_actions))


    stuff = list(map(
                lambda i: (i, scores, num_bandits, num_agents, epsilons, num_actions, non_stationary),
                list(range(num_experiments))
            ))

    global NUM_THREADS_LEFT
    NUM_THREADS_LEFT = num_experiments
    results = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=AVAIL_THREADS) 
    for el in stuff:
        res = executor.submit(experiment_thread,el)
        results.append(res)

    executor.shutdown(True)
    for res in results:
        res.result()

    score_means = scores.mean(0)

    # with open("scores.txt", 'w') as f:
    #     f.write(
    # )


    fig, ax = plt.subplots()
    for i in range(num_agents):
        eps = epsilons[i]
        ax.plot(score_means[i,:], label=f"{eps=}")
    ax.legend()
    title = f"Experimnts: {num_experiments}"
    if non_stationary:
        title += "\nNon-stationary"

    if LR is not None:
        title += f"\n{LR=}"
    else:
        title += "Average Q"

    fig.suptitle(title)

    plt.show()


main()
        







# Cell
