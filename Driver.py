"""module docstring
"""

import Jarvis
import numpy as np

# Import training data
X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
Y = np.array(([75], [82], [93]), dtype=float)

# Normalize training data
X_max = np.amax(X, axis=0)
X = X/X_max
Y = Y/100

# Import test data
X_test = np.array(([8, 3]), dtype=float)

# Normalize test data
X_test = X_test/X_max

# Initialize Neural Network
NN = Jarvis.Neuron()

# Initialize training iterations
incrementer = 0.5
loop_num = 1
cost = [None]*(loop_num + 1)

# Run training iterations
for iteration in range(0, loop_num):
    cost[iteration] = NN.cost_function(X, Y)
    dJdW1, dJdW2 = NN.cost_function_prime(X, Y)
    NN.W1 = NN.W1 - incrementer*dJdW1
    NN.W2 = NN.W2 - incrementer*dJdW2

print(dJdW1, dJdW2)