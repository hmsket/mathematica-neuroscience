import random
import copy
import matplotlib.pyplot as plt

def sgn(u):
    if u > 0:
        return 1
    else:
        return -1

def generate_memories(m, n):
    memories = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            rnd = random.random() # rnd ~ U(0,1)
            if rnd > 0.5:
                memories[i][j] = 1
            else:
                memories[i][j] = -1
    return memories

def set_weights(memories, m, n):
    weights = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            sum = 0
            for k in range(m):
                sum += memories[k][i] * memories[k][j]
            w = sum / n
            weights[i][j] = w
            weights[j][i] = w
    return weights

def set_initial_state(memory, alpha=0):
    state = copy.deepcopy(memory)
    for i in range(alpha):
        state[i] = memory[alpha-(i+1)]
    return state

def update_state(weights, state, n):
    next_state = [0 for _ in range(n)]
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += weights[i][j] * state[j]
        next_state[i] = sgn(sum)
    return next_state

def calc_direction_cosine(pattern, state, n):
    dc = 0
    for i in range(n):
        dc += pattern[i] * state[i]
    dc /= n
    return dc

def main(m=80, n=1000, N=19):
    memories = generate_memories(m, n)
    weights = set_weights(memories, m, n)
    alphas = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for alpha in alphas:
        state = set_initial_state(memories[0], alpha)
        dc = calc_direction_cosine(memories[0], state, n)
        dcs = []
        dcs.append(dc)
        for _ in range(N):
            state = update_state(weights, state, n)
            dc = calc_direction_cosine(memories[0], state, n)
            dcs.append(dc)
        plt.plot(range(N+1), dcs, marker=".", label=alpha)
    plt.xlabel("TIME")
    plt.ylabel("DIRECTION COSINE")
    plt.savefig("dynamic.png")
    #plt.legend(title="alpha")
    plt.show()

if __name__ == "__main__":
   main()
