# Cell
from bandit import Bandits
import numpy as np
import random
from threading import Lock
import matplotlib.pyplot as plt


LR = 0.35
# LR = None
# USE_LINSPACE_MEANS = True
USE_LINSPACE_MEANS = False
NON_STATIONARY = True

mutex = Lock()


class Agent:
    def __init__(self, num_bandits, epsilon, lr=None):
        self.num_bandits = num_bandits
        self.qs = np.zeros(num_bandits) + 2
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
    bandit = Bandits(num_bandits, means, None, non_stationary)
    if epsilons is None:
        epsilons = np.linspace(0, 0.15, num_agents)

    agents = list(map(lambda x: Agent(num_bandits, x,LR), epsilons))

    scores = np.zeros((num_agents, num_actions))
    optimal_actions = np.zeros((num_agents, num_actions)) # 0 or 1 - can then take avg

    for n in range(num_actions):
        for i,agent in enumerate(agents):
            optimal_move = np.argmax(bandit.means)
            move = agent.get_action()
            reward = bandit.pull(move)
            agent.update(reward, move)
            scores[i,n] = reward
            if move == optimal_move:
                optimal_actions[i, n] = 1


    return scores, optimal_actions

def main():
    num_agents = 5
    num_bandits = 10
    epsilons = [0,0.001, 0.01, 0.05, 0.1]
    num_actions = 2000
    num_experiments = 300
    non_stationary = NON_STATIONARY

    scores = np.zeros((num_experiments, num_agents, num_actions))
    optimal = np.zeros((num_experiments, num_agents, num_actions))


    stuff = list(map(
                lambda i: (i, scores, optimal, num_bandits, num_agents, epsilons, num_actions, non_stationary),
                list(range(num_experiments))
            ))

    for i in range(num_experiments):
        if(i % 10 == 0):
            print(f"Iteration {i}/{num_experiments}")
        s, o = experiment(num_bandits, num_agents, epsilons, num_actions, non_stationary)
        scores[i,:,:] = s
        optimal[i,:,:] = o
 

    score_means = scores.mean(0)
    optimal_perc = optimal.mean(0)

    fig, ax = plt.subplots(1, 2)
    for i in range(num_agents):
        eps = epsilons[i]
        ax[0].plot(score_means[i,:], label=f"{eps=}")
        ax[1].plot(optimal_perc[i,:], label=f"{eps=}")
    ax[1].set_ylim(0, 1)
    ax[0].set_title("Average score")
    ax[1].set_title("Average optimal %")
    ax[0].legend();ax[1].legend()
    title = f"Experimnts: {num_experiments}"
    if non_stationary:
        title += "\nNon-stationary"

    if LR is not None:
        title += f"\n{LR=}"
    else:
        title += "\nAverage Q"

    fig.suptitle(title)
    plt.show()


main()
        







# Cell
