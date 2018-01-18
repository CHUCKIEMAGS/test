import numpy as np
import YTExample

class Brain(object):
    
    #Initializations
    def __init__(self):
        #Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize1 = 3

        #Initialize regular parameters
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize1)
        self.W2 = np.random.randn(self.hiddenLayerSize1,self.outputLayerSize)

    #Forward propogation
    def forward(self, X):
        #Forward propogation of inputs through NN
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.actFunc(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.actFunc(self.z3)
        return yHat

    def actFunc(self, z):
        #Sigmoid function for neuron activation
        return 1/(1+np.exp(-z))
    
    #Backpropogation
    def actFuncPrime(self, z):
        #Gradient of sigmoid function
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, X, y):
        #Error function    
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self, X, y):
        #Derivative of cost function with respect to W and W2 for a given X and y
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.actFuncPrime(self.z3))
        dJdW2 = np.dot(self.a2.T,delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.actFuncPrime(self.z2)
        dJdW1 = np.dot(X.T,delta2)

        return dJdW1, dJdW2
