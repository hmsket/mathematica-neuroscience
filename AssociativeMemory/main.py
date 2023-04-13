import numpy as np
import matplotlib.pyplot as plt

def sgn(u):
    u = np.where(u>0, 1, -1)
    return u

def generate_memories(m, n):
    rng = np.random.default_rng()
    memories = rng.choice([-1,1], p=[0.5,0.5], size=(m,n))
    return memories

def set_weights(memories, n, mu=0.08):
    tmp_weights = np.dot(memories.T, memories)
    weights = mu * tmp_weights / n
    # おもみ行列の対角成分を0にする
    v = np.zeros(n)
    np.fill_diagonal(weights, v)
    return weights

def set_initial_state(memory, alpha):
    state = np.copy(memory)
    for i in range(alpha):
        state[i] = memory[alpha-(i+1)]
    return state

def update_state(weights, state):
    tmp_next_state = np.dot(weights, state)
    next_state = sgn(tmp_next_state)
    return next_state

def calc_direction_cosine(pattern, state, n):
    tmp_dc = np.dot(pattern, state)
    dc = tmp_dc / n
    return dc

def main(m=80, n=1000, N=30, step=11):
    memories = generate_memories(m, n)
    weights = set_weights(memories, n)
    alphas = np.linspace(0, n, step, dtype=int) # e.g. [0, 100, 200, ..., 900, 1000]
    for alpha in alphas:
        dcs = []
        state = set_initial_state(memories[0], alpha)
        dc = calc_direction_cosine(memories[0], state, n)
        dcs.append(dc)
        for _ in range(N):
            state = update_state(weights, state)
            dc = calc_direction_cosine(memories[0], state, n)
            dcs.append(dc)
        plt.plot(range(N+1), dcs, marker=".", label=alpha)
    plt.title(f"$m={m},n={n}$")
    plt.xlabel("TIME")
    plt.ylabel("DIRECTION COSINE")
    plt.savefig("cosine.png")
    #plt.legend(title="alpha")
    #plt.show()

if __name__ == "__main__":
    main()
