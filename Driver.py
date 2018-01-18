from Brain1 import Brain
from YTExample import X, y 

alpha = 3

NN = Brain()



for iterator in range(0, 1000):
    dJdW1, dJdW2 = NN.costFunctionPrime(X, y)
    NN.W1 = NN.W1 - alpha*dJdW1
    NN.W2 = NN.W2 - alpha*dJdW2
    print(NN.costFunction(X, y))

print(NN.yHat)
print(y)