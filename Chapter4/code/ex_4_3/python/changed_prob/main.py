from concurrent.futures import thread
import numpy as np
from scipy.stats import poisson
from collections import defaultdict as dd
import multiprocessing
from multiprocessing import Pool, Lock
from utils import CompleteTaskCounter
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import  seaborn as sb

NUM_THREADS = multiprocessing.cpu_count()
thread_counter = None

MAX_CARS = 20
MAX_MOVE = 5

MOVE_REW = -2
RENT_REW = 10
ADDITIONAL_PARKING_REW = -4

L_FST_REQ = 3
L_SND_REQ = 4
L_FST_RET = 3
L_SND_RET = 2

GAMMA = 0.9

P_PLOT_CNTR = 0
V_PLOT_CNTR = 0


dynamics_cache = {}

cache_lock = Lock()

def all_states():
    for i in range(MAX_CARS + 1):
        for j in range(MAX_CARS + 1):
            yield i, j

def available_actions(state):
    fst, snd = state
    # how many can we move from snd to first
    lower_bound = max(-snd, fst - MAX_CARS, -MAX_MOVE)

    # One car for free from fst to snd, so we add one to max move here
    upper_bound = min(fst, MAX_CARS - snd, MAX_MOVE + 1)

    for i in range(lower_bound, upper_bound + 1, 1):
        yield i


def prob(val, lam, max):
    # if the value is at a max, we return the rest of the cdf
    if val == max:
        return 1 - poisson.cdf(val - 1, lam)
    else:
        return poisson.pmf(val, lam)

def compute_dynamics(state):
    # assumes happens after action
    fst, snd = state[0] , state[1]
    state_rew = dd(int)

    for fst_req in range(fst + 1): # number of requests in fst
        fst_left = fst - fst_req
        # number of returns in fst
        # we can return MAX_CARS(fst - fst_req)
        p = prob(fst_req, L_FST_REQ, fst)
        for fst_ret in range(MAX_CARS - fst_left + 1):
            p2 = p * prob(fst_ret, L_FST_RET, MAX_CARS - fst_left)

            for snd_req in range(snd + 1): # number of request in snd
                snd_left = snd - snd_req
                p3 = p2 * prob(snd_req, L_SND_REQ, snd)

                # the number of returns in snd
                for snd_ret in range(MAX_CARS - snd_left + 1):
                    p4 = p3 * prob(snd_ret, L_SND_RET, MAX_CARS - snd_left)

                    reward = (fst_req + snd_req) * RENT_REW 
                    state = (fst_left + fst_ret, snd_left + snd_ret)
                    state_rew[state, reward] += p4
    return state_rew


def dynamics(state, action):
    # 2$ cost of moving each car
    if isinstance(action, float):
        raise RuntimeError(f'action must be int: {action}')

    move_rew = abs(action) * MOVE_REW

    # one car for free from fst to snd
    if action > 0:
        move_rew - MOVE_REW

    newstate = state[0] -action, state[1] + action

    # compute the cars of the additional cars - 4$ if over 10
    park_rew = 0
    if newstate[0] > 10:
        park_rew += ADDITIONAL_PARKING_REW
    if newstate[1] > 10:
        park_rew += ADDITIONAL_PARKING_REW

    if (newstate) in dynamics_cache:
        cache_lock.acquire()
        val = dynamics_cache[newstate]
        cache_lock.release()
        for state_rew, p in val.items():

            state, reward = state_rew

            yield state, reward + move_rew + park_rew, p
    else:
        raise RuntimeError(f"State not in cache: {newstate}")
    

def thread_dyn(args):
    c, s= args
    global thread_counter
    state_rew = compute_dynamics(s)
    thread_counter.incrprint(c)
    return s, state_rew

def policy_eval_thread(args):
    c, policy, values, fst, snd = args
    
    v = values[fst][snd]
    newval = 0

    if(isinstance(policy[fst][snd], float)):
        raise RuntimeError(f'policy must be int {policy[fst][snd]}')

    for state, reward, p in dynamics((fst, snd), policy[fst][snd]):
        newval += p*(reward + GAMMA * values[state[0]][state[1]])
    delta = abs(v - newval)

    global thread_counter
    thread_counter.incrprint(c)

    return newval, fst, snd, delta




def policy_eval(policy, values):
    eps = 10e-1
    delta = 0
    iters = 0
    changed = True

    while delta > eps or iters == 0:
        delta = 0
        global thread_counter
        thread_counter = CompleteTaskCounter((MAX_CARS + 1) ** 2)
        def args():
            for fst, snd in all_states():
                yield(thread_counter.count, policy,values, fst, snd)

        with Pool(NUM_THREADS) as p:
            res = p.map(policy_eval_thread, args())
        for newval, fst, snd, d in res:
            values[fst][snd] = newval
            delta = max(delta, d)


        if delta <= eps and iters == 0:
            changed = False
        iters += 1
        print(f'{iters=}')

    return changed

def policy_plot(policy):
    ax = sb.heatmap(policy, vmin=-MAX_MOVE, vmax=MAX_MOVE + 1, center=0, annot=True)
    ax.invert_yaxis()
    global P_PLOT_CNTR
    name = f"policy{P_PLOT_CNTR}"
    plt.title(name)
    plt.savefig(name)
    P_PLOT_CNTR += 1
    plt.close()



def value_plot(value):
    fig, ax = plt.subplots(figsize=(10, 10))
    cs = ax.contour(range(0, MAX_CARS + 1), range(0, MAX_CARS +1), value)
    ax.clabel(cs)
    global V_PLOT_CNTR
    name = f"values{V_PLOT_CNTR}"
    ax.set_title(name)
    fig.savefig(name)
    V_PLOT_CNTR += 1
    plt.close()
    # plt.show()


def pol_improve_thread(args):
    c, policy, values, fst, snd = args
    policy_stable = True
    old_action = policy[fst][snd]
    bst = -1000000
    bst_action = old_action
    for action in available_actions((fst, snd)):
        val = 0
        for state, reward, p in dynamics((fst, snd), action):
            val += p*(reward + GAMMA * values[state[0]][state[1]])
        # print(f'{action=} : {val=}')

        if val > bst:
            bst = val
            bst_action = action

    # print(f'{bst_action=} : {bst=}')

    if old_action != bst_action:
        policy_stable = False

    global thread_counter
    thread_counter.incrprint(c)
    # print(f"before - {policy_stable=} {bst_action=} {fst=} {snd=}")
    return policy_stable, bst_action, fst, snd

def pol_improve(policy, values):
    res = None
    global thread_counter
    thread_counter = CompleteTaskCounter((MAX_CARS + 1) ** 2)
    def args():
        for fst, snd in all_states():
            yield (thread_counter.count, policy, values, fst, snd)

    res = None
    with Pool(NUM_THREADS) as p:
        res = p.map(pol_improve_thread, args())

    policy_stable = True
    for ps, bst_action, fst, snd in res:
        # print(f"res - {ps=} {bst_action=} {fst=} {snd=}")
        policy_stable = ps and policy_stable
        policy[fst][snd] = bst_action

    return policy_stable


def main():
    policy = np.zeros(( MAX_CARS + 1, MAX_CARS + 1), dtype=int)
    values = np.zeros(( MAX_CARS + 1, MAX_CARS + 1))

    global dynamics_cache
    if "cache" in os.listdir("./"):
        f = open("cache", "rb")
        dynamics_cache = pickle.load(f)
        f.close()

    else:
        print("COMPUTING THE DYNAMICS - WILL CACHE")
        global thread_counter
        thread_counter = CompleteTaskCounter((MAX_CARS + 1) **2 )
        def arg_iterator():
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                        yield thread_counter.count, (i, j)

        ret = None
        with Pool(NUM_THREADS) as p:
            ret = p.map(thread_dyn, arg_iterator())

        for s, sr in ret:
            dynamics_cache[s] = sr
        
        f = open("cache", "wb")
        pickle.dump(dynamics_cache, f)
        f.close()

    stable = False
    while not stable:
        print("POLICY EVALUATION")
        changed = policy_eval(policy, values)    
        value_plot(values)
        if not changed:
            break
        print("POLICY IMPROVEMENT")
        stable = pol_improve(policy, values)
        policy_plot(policy)
    

main()
    
        
        







