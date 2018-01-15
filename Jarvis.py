"""module docstring
"""

import numpy as np

class Neuron():
    def __init__(self):
        # Neural net parameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Neuron weights
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
    
    def forward(self, X):
        # Forward propagation function  
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.activation_function(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.activation_function(self.z3)
        return yHat

    def activation_function(self, z):
        # Sigmoid function
        return 1 / (1 + np.exp(-1 * z))

    def activation_function_prime(self,z):
        # Derivative of sigmoid function
        prime = self.activation_function(z)
        prime_output = prime*(1 - prime)
        return prime_output

    def cost_function(self, X, Y):
        self.yHat = self.forward(X)
        J = 0.5*sum((Y-self.yHat)**2)
        return J 

    def cost_function_prime(self, X, Y):
       self.yHat = self.forward(X)

       delta3 = np.multiply(-(Y-self.yHat), self.activation_function_prime(self.z3))
       dJdW2 = np.dot(self.a2.T, delta3)     

       delta2 = np.dot(delta3, self.W2.T)*self.activation_function_prime(self.z2)
       dJdW1 = np.dot(X.T, delta2)

       return dJdW1, dJdW2

    def inverse_Hessian(self, M, w_t, w_t2, grad_1, grad_2):










    def rho_init(dJdW1, dJdW2)       

    def BFGS(self, X, Y):
        grad_1, grad_2 = self.cost_function_prime(X, Y)
        m_1, n_1 = np.shape(grad_1)
        m_2, n_2 = np.shape(grad_2)

