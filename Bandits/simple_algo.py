# Cell
import numpy as np
import random
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool, Manager, Lock

from utils import print_prog_bar, CompleteTaskCounter
from agents import EpsilonAgent
from bandit import Bandits

NUM_THREADS = multiprocessing.cpu_count()


# Lr can be a lambda of n, None for avg, or constant
LR = 0.2
NUM_AGENTS = 5
NUM_BANDITS = 10
EPSILONS = [0,0.001, 0.01, 0.05, 0.1]
NON_STATIONARY = True
NUM_ACTIONS = 1000
NUM_EXPERIMENTS = 1000

# thread_count_mutex = Lock()

def experiment():

    # Initialize with default means and stevs
    bandit = Bandits(NUM_BANDITS, None, None, NON_STATIONARY)

    agents = list(map(lambda x: EpsilonAgent(NUM_BANDITS, x,LR), EPSILONS))

    scores = np.zeros((NUM_AGENTS, NUM_ACTIONS))
    optimal_actions = np.zeros((NUM_AGENTS, NUM_ACTIONS)) # 0 or 1 - can then take avg

    for n in range(NUM_ACTIONS):
        for i,agent in enumerate(agents):
            optimal_move = np.argmax(bandit.means)
            move = agent.get_action()
            reward = bandit.pull(move)
            agent.update(reward, move)
            scores[i,n] = reward
            if move == optimal_move:
                optimal_actions[i, n] = 1


    return scores, optimal_actions

def thread_experiment(counter):
    out =  experiment()


    counter.incrprint()
    # with thread_count_mutex:
    #     # decrement the number left and 
    #     completed.value += 1
    #     print_prog_bar(completed.get(), NUM_EXPERIMENTS)

    return out

def main():

    scores = np.zeros((NUM_EXPERIMENTS, NUM_AGENTS, NUM_ACTIONS))
    optimal = np.zeros((NUM_EXPERIMENTS, NUM_AGENTS, NUM_ACTIONS))

    # m = Manager()

    # num_threads_done = m.Value('i', 0)
    thread_counter = CompleteTaskCounter(NUM_EXPERIMENTS)

    res = None
    with Pool(NUM_THREADS) as p:
        # res = p.map(thread_experiment, (num_threads_done for _ in range(NUM_EXPERIMENTS)))
        res = p.map(thread_experiment, (thread_counter for _ in range(NUM_EXPERIMENTS)))

    for i, el in enumerate(res):
        s, o = el
        scores[i,:,:] = s
        optimal[i,:,:] = o

    score_means = scores.mean(0)
    optimal_perc = optimal.mean(0)

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    for i in range(NUM_AGENTS):
        eps = EPSILONS[i]
        ax[0].plot(score_means[i,:], label=f"{eps=}")
        ax[1].plot(optimal_perc[i,:], label=f"{eps=}")
    ax[1].set_ylim(0, 1)
    ax[0].set_title("Average score")
    ax[1].set_title("Average optimal %")
    ax[0].legend();ax[1].legend()
    title = f"Experiments: {NUM_EXPERIMENTS}"
    if NON_STATIONARY:
        title += "\nNon-stationary"

    if LR is not None:
        title += f"\n{LR=}"
    else:
        title += "\nAverage Q"

    fig.suptitle(title)

    plt.savefig(f"./figures/exmnts{NUM_EXPERIMENTS}_lr{LR}_{'nonstat' if NON_STATIONARY else 'stat'}_agnts{NUM_AGENTS}_actions{NUM_ACTIONS}.png")
    plt.show()


main()
        







# Cell
