import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from utils import CompleteTaskCounter
from agents import EpsilonAgent, GradientAgent, UCBAgent
from bandit import Bandits
from enum import Enum
from itertools import repeat

class AgentTypes(Enum):
    EpsilonGreedyMean=1
    EpsilonGreedyConst=2
    OptInitVals=3 # this agent should be greedy
    UCB=4
    GradientAgent=5



NUM_THREADS = multiprocessing.cpu_count()

# Experiment specification
E_NUM_AGENT_TYPES = 5
E_NUM_ACTIONS = 10000
E_NUM_EXPERIMENTS = 2000
E_NUM_PARAMETER_VALS = 6

# Agent specification

# Make function to avoid polluting namespace
def get_expspace(lower, upper, length):
    return np.exp(
        np.linspace(
            np.log(lower),
            np.log(upper),
            length
        )
    )

# Lr can be a lambda of n, None for avg, or constant
# epsilon greedy

# All epsilon-greedy agents have this learning rate
A_LR_DEFAULT = 0.1
A_EPS_INIT_Q = 0.5 # Not optimistic

# mean agent
A_EPS_MEAN_EPS = get_expspace(1/128, 1/4, E_NUM_PARAMETER_VALS)

# constant learning rate agent
A_EPS_CONST_EPS = get_expspace(1/64, 1/2, E_NUM_PARAMETER_VALS)

# Optimal initial values
A_EPS_OPT_QS = get_expspace(2.2, 14, E_NUM_PARAMETER_VALS)

# UCB agent
A_UCB_CS = get_expspace(1/16, 4, E_NUM_PARAMETER_VALS)

# Gradient agent
A_GRAD_LRS = get_expspace(1/32, 4, E_NUM_PARAMETER_VALS)


# Bandit specification
B_NUM_BANDITS = 10
B_NON_STATIONARY = True
B_RAND_WALK_STD = 0.01
B_MEANS_MEAN = 2
B_MEANS_STD = 1
B_BANDIT_STD = 0.2

# has to be global
thread_counter = None


def get_agent_name(agent_type):
    if agent_type == AgentTypes.EpsilonGreedyMean:
        return 'e-greedy mean'
    if agent_type == AgentTypes.EpsilonGreedyConst:
        return 'e-greedy const'
    if agent_type == AgentTypes.OptInitVals:
        return 'greedy oiv'
    if agent_type == AgentTypes.UCB:
        return 'UCB'
    if agent_type == AgentTypes.GradientAgent:
        return 'Gradient agent'

def get_agent_pars(agent_type):
    if agent_type == AgentTypes.EpsilonGreedyMean:
        return A_EPS_MEAN_EPS
    if agent_type == AgentTypes.EpsilonGreedyConst:
        return A_EPS_CONST_EPS
    if agent_type == AgentTypes.OptInitVals:
        return A_EPS_OPT_QS
    if agent_type == AgentTypes.UCB:
        return A_UCB_CS
    if agent_type == AgentTypes.GradientAgent:
        return A_GRAD_LRS

         
def make_agent(agent_type, par_val):
    if agent_type == AgentTypes.EpsilonGreedyMean:
        return EpsilonAgent(
            B_NUM_BANDITS,
            par_val,
            A_EPS_INIT_Q,
            None # will get mean agent
        )
    if agent_type == AgentTypes.EpsilonGreedyConst:
        return EpsilonAgent(
            B_NUM_BANDITS,
            par_val,
            A_EPS_INIT_Q,
            A_LR_DEFAULT
        )

    if agent_type == AgentTypes.OptInitVals:
        return EpsilonAgent(
            B_NUM_BANDITS,
            0,
            par_val,
            A_LR_DEFAULT
        )

    if agent_type == AgentTypes.UCB:
        return UCBAgent(
            B_NUM_BANDITS,
            A_EPS_INIT_Q,
            par_val,
            A_LR_DEFAULT
        )

    if agent_type == AgentTypes.GradientAgent:
        return GradientAgent(
            B_NUM_BANDITS,
            par_val
        )

 
    

def single_run(agent_type, parameter_val):

    # Initialize with default means and stevs
    bandit = Bandits(
        B_NUM_BANDITS, 
        B_MEANS_MEAN,
        B_MEANS_STD,
        B_BANDIT_STD,
        B_NON_STATIONARY,
        B_RAND_WALK_STD
    )

    scores = np.zeros(E_NUM_ACTIONS)

    agent = make_agent(agent_type, parameter_val)

    for n in range(E_NUM_ACTIONS):
        move = agent.get_action()
        reward = bandit.pull(move)
        agent.update(reward, move)
        scores[n] = reward

    return scores.mean()

def thread_run(args):
    global thread_counter
    out = single_run(*(args[1:]))
    thread_counter.incrprint(args[0])
    return out

def experiment(agent_type):
    # We need to run through the parameter values
    # and get NUM_EXPERIMENTS values 
    value_means = np.zeros(E_NUM_PARAMETER_VALS)
    print(get_agent_name(agent_type))
    global thread_counter
    thread_counter = CompleteTaskCounter(E_NUM_PARAMETER_VALS * E_NUM_EXPERIMENTS)

    for i,pval in enumerate(get_agent_pars(agent_type)):
        args = (thread_counter.count, agent_type, pval)
        with Pool(NUM_THREADS) as p:
            sms = p.map(thread_run, repeat(args, E_NUM_EXPERIMENTS))
        value_means[i] = np.mean(sms)

    # return list of parameter values; the indices should correspond to 
    # the ones defined at the top of the file

    return value_means

def make_plot(means):

    fig,ax= plt.subplots(1, 1,figsize=(10, 7))
    for ag, means in means.items():

        ax.plot(
            get_agent_pars(ag), 
            means, 
            label=get_agent_name(ag),
            linewidth=4
        )
    ax.legend()
    ax.set_ylabel(f"Average reward over {E_NUM_ACTIONS} steps")
    ax.set_xscale('log')
    ax.set_xlim(1/128, 10)

    ticks = [
        f'1/{2 ** (-i)}' if i < 0 else ('1' if i == 0 else f"{2**i}")
        for i in range(-7, 5)
    ]

    ax.set_xticks(ticks=list(map(lambda x: float(eval(x)), ticks)))
    ax.set_xticklabels(ticks)
    plt.plot()
    plt.savefig(
        f"COMPARISONexpnts{E_NUM_EXPERIMENTS}_numsteps{E_NUM_ACTIONS}_pvals{E_NUM_PARAMETER_VALS}.png"
    )
    plt.show()
    
    
def main():
    means = {}
    for ag in AgentTypes:
        means[ag] = experiment(ag)
    make_plot(means)


if __name__ == "__main__":
    main()
