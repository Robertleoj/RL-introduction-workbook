import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from utils import CompleteTaskCounter
from agents import EpsilonAgent
from bandit import Bandits

NUM_THREADS = multiprocessing.cpu_count()

# Experiment specification
E_NUM_AGENTS = 3
E_NUM_ACTIONS = 2000
E_NUM_EXPERIMENTS = 2000

# Agent specification
# Lr can be a lambda of n, None for avg, or constant
A_LR = 0.2
A_EPSILONS = [0, 0.01, 0.1]
A_INITIAL_Q = 0

# Bandit specification
B_NUM_BANDITS = 10
B_NON_STATIONARY = True
B_RAND_WALK_STD = 0.01
B_MEANS_MEAN = 2
B_MEANS_STD = 1
B_BANDIT_STD = 0.2


# has to be global
thread_counter = None

def experiment():

    # Initialize with default means and stevs
    bandit = Bandits(
        B_NUM_BANDITS, 
        B_MEANS_MEAN,
        B_MEANS_STD,
        B_BANDIT_STD,
        B_NON_STATIONARY,
        B_RAND_WALK_STD
    )

    agents = list(map(
        lambda eps: EpsilonAgent(
            B_NUM_BANDITS, 
            eps, 
            A_INITIAL_Q ,
            A_LR
        ), 
        A_EPSILONS
    ))

    scores = np.zeros((E_NUM_AGENTS, E_NUM_ACTIONS))
    optimal_actions = np.zeros((E_NUM_AGENTS, E_NUM_ACTIONS)) # 0 or 1 - can then take avg

    for n in range(E_NUM_ACTIONS):
        for i,agent in enumerate(agents):
            optimal_move = np.argmax(bandit.means)
            move = agent.get_action()
            reward = bandit.pull(move)
            agent.update(reward, move)
            scores[i,n] = reward
            if move == optimal_move:
                optimal_actions[i, n] = 1


    return scores, optimal_actions

def thread_experiment(counterval):
    out =  experiment()

    # see progress
    thread_counter.incrprint(counterval)

    return out

def plot_rewards_and_percentage(scores, optimal):

    score_means = scores.mean(0)
    optimal_perc = optimal.mean(0)

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    for i in range(E_NUM_AGENTS):
        eps = A_EPSILONS[i]
        ax[0].plot(score_means[i,:], label=f"{eps=}")
        ax[1].plot(optimal_perc[i,:], label=f"{eps=}")
    ax[1].set_ylim(0, 1)
    ax[0].set_title("Average score")
    ax[1].set_title("Average optimal %")
    ax[0].legend();ax[1].legend()
    title = f"Experiments: {E_NUM_EXPERIMENTS}"
    if B_NON_STATIONARY:
        title += "\nNon-stationary"

    if A_LR is not None:
        title += f"\n{A_LR=}"
    else:
        title += "\nAverage Q"

    fig.suptitle(title)

    plt.savefig(
        f"./figures/EPSILONAGENT_exmnts{E_NUM_EXPERIMENTS}_lr{A_LR}_{'nonstat' if B_NON_STATIONARY else 'stat'}_agnts{E_NUM_AGENTS}_actions{E_NUM_ACTIONS}_initq{A_INITIAL_Q}.png"
    )
    plt.show()



def main():

    scores = np.zeros((E_NUM_EXPERIMENTS, E_NUM_AGENTS, E_NUM_ACTIONS))
    optimal = np.zeros((E_NUM_EXPERIMENTS, E_NUM_AGENTS, E_NUM_ACTIONS))

    global thread_counter
    thread_counter = CompleteTaskCounter(E_NUM_EXPERIMENTS)

    res = None
    with Pool(NUM_THREADS) as p:
       res = p.map(thread_experiment, thread_counter.rep_gen())

    for i, el in enumerate(res):
        s, o = el
        scores[i,:,:] = s
        optimal[i,:,:] = o

    plot_rewards_and_percentage(scores, optimal)

if __name__ == "__main__":
    main()
