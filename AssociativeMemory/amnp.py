import numpy as np
import copy
import matplotlib.pyplot as plt

def sgn(u):
    for i in range(len(u)):
        if u[i] > 0:
            u[i] = 1
        else:
            u[i] = -1
    return u

def generate_memories(m, n):
    memories = np.random.choice([-1,1], size=(m,n), p=[0.5,0.5])
    return memories

def set_weights(memories, n, mu=0.08):
    weights = np.dot(memories.T, memories)
    weights = mu * weights / n
    v = np.zeros(n)
    np.fill_diagonal(weights, v) # 対角成分を0にする
    return weights

def set_initial_state(memory, alpha):
    state = copy.deepcopy(memory)
    for i in range(alpha):
        state[i] = memory[alpha-(i+1)]
    return state

def update_state(weights, state):
    next_state = np.dot(weights, state)
    next_state = sgn(next_state)
    return next_state

def calc_direction_cosine(pattern, state, n):
    dc = np.dot(pattern, state)
    dc /= n
    return dc

def main(m=80, n=1000, N=30, step=11):
    np.random.seed(20220622)
    memories = generate_memories(m, n)
    weights = set_weights(memories, n)
    alphas = np.linspace(0, n, step, dtype=int) # e.g. [0, 100, 200, ..., 900, 1000]
    for alpha in alphas:
        state = set_initial_state(memories[0], alpha)
        dc = calc_direction_cosine(memories[0], state, n)
        dcs = []
        dcs.append(dc)
        for _ in range(N):
            state = update_state(weights, state)
            dc = calc_direction_cosine(memories[0], state, n)
            dcs.append(dc)
        plt.plot(range(N+1), dcs, marker=".", label=alpha)
    plt.title(f"$m={m},n={n}$")
    plt.xlabel("TIME")
    plt.ylabel("DIRECTION COSINE")
    plt.savefig(f"m{m}.png")
    #plt.legend(title="alpha")
    #plt.show()

if __name__ == "__main__":
    main()