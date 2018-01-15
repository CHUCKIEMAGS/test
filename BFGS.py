import numpy as np
import Jarvis
X = np.array(([.3, .5], [.5, .1], [1, .2]), dtype=float)
Y = np.array(([.75], [.82], [.93]), dtype=float)


def cf_gradient_calc(self, X, Y):
    self.yHat = self.forward(X)

    delta3 = np.multiply(-(Y-self.yHat), self.activation_function_prime(self.z3))
    dJdW2 = np.dot(self.a2.T, delta3)     

    delta2 = np.dot(delta3, self.W2.T)*self.activation_function_prime(self.z2)
    dJdW1 = np.dot(X.T, delta2)

    return dJdW1, dJdW2

def BFGS(self, X, Y):
    grad_1, grad_2 = self.cf_gradient_calc(X, Y)
    m_1, n_1 = np.shape(grad_1)
    m_2, n_2 = np.shape(grad_2)    