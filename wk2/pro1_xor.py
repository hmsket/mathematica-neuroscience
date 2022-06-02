import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1/(1+np.exp(-x))

def squared_error(x, t):
  return (x-t)**2 / 2

np.random.seed(20220525)

lr = 0.8
I, H, O = 3, 2, 1

inputs = np.array(([1,0,0],[1,0,1],[1,1,0],[1,1,1]))
hiddens = np.zeros(H)
output = np.zeros(O)
teachers = np.array((0,1,1,0))

S = np.random.normal(0.0, 0.2, (I, H))
W = np.random.normal(0.0, 0.2, (H+1, O))

errors = []

N = 10000
for i in range(N):
  rnd = np.random.randint(0,4)
  input = inputs[rnd]
  teacher = teachers[rnd]
  hiddens = np.dot(input, S)
  hiddens = sigmoid(hiddens)
  hiddens_tmp = np.concatenate([[1], hiddens])
  output = np.dot(hiddens_tmp, W)
  output = sigmoid(output)
  error = squared_error(output, teacher)
  r = (teacher-output) * output * (1-output)
  delta_W = lr * r * hiddens_tmp
  r_dash = r * W[1:].T * hiddens * (1-hiddens)
  delta_S = lr * r_dash.T * input
  W += delta_W.reshape(H+1, O)
  S += delta_S.reshape(I, H)
  errors.append(error)

# 検証
for i in range(len(inputs)):
  input = inputs[i]
  teacher = teachers[i]
  hiddens = np.dot(input, S)
  hiddens = sigmoid(hiddens)
  hiddens_tmp = np.concatenate([[1], hiddens])
  output = np.dot(hiddens_tmp, W)
  output = sigmoid(output)
  print(f"input: {input}, output: {output}, teacher: {teacher}")

plt.plot(errors, linestyle="None", marker=".", markersize="3")
plt.grid(True)
plt.xlabel('iterations')
plt.ylabel('E')
plt.savefig('xor_E_h2.png')
#plt.show()