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

def set_initial_state(memory, alpha=200):
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

def count_error(pattern, state, n):
    count = 0
    for i in range(n):
        if pattern[i] != state[i]:
            count += 1
    return count

def main():
    m = 80
    n = 1000
    memories = generate_memories(m, n)
    weights = set_weights(memories, m, n)
    state = set_initial_state(memories[0])
    N = 100
    for _ in range(N):
        print(count_error(memories[0], state, n))
        state = update_state(weights, state, n)
    print(count_error(memories[0], state, n))

if __name__ == "__main__":
   main()
