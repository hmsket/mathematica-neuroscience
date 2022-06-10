import random
import copy

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
            if rnd > 0.5: # > か >= かは迷う
                memories[i][j] = 1
            else:
                memories[i][j] = -1
    return copy.deepcopy(memories)

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

def set_initial_state(memory, n, alpha=4):
    tmp = copy.deepcopy(memory)
    for i in range(alpha):
        memory[i] = tmp[alpha-(i+1)]
    network = [0 for _ in range(n)]
    for i in range(n):
        network[i] = memory[i]
    return network

def update_state(weights, network, n):
    tmp = copy.deepcopy(network)
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += weights[i][j] * tmp[j]
        network[i] = sgn(sum)
    return network

def error(pattern, network, n):
    count = 0
    for i in range(n):
        if pattern[i] != network[i]:
            count += 1
    return count

if __name__ == "__main__":
    m = 8
    n = 10
    memories = generate_memories(m, n)
    weights = set_weights(memories, m, n)
    network = set_initial_state(memories[0], n)
    N = 100
    for i in range(N):
        print(error(memories[0], network, n))
        network = update_state(weights, network, n)
    print(error(memories[0], network, n))
