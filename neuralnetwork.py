import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

# create a simple neural network
nn = NeuralNetwork(2, 3, 1)
print("Input to Hidden Weights:")
print(nn.W1)
print("\nHidden to Output Weights")
print(nn.W2)

"""
Activation Functions
  Activation functions introduce non-linearity into the network, allowing it to learn complex patterns.
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# testing sigmoid function
x = np.linspace(-10, 10, 100)
y = sigmoid(x)

import matplotlib.pyplot as plt

plt.plot(x, y)
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.grid(True)
plt.show()

"""
Forward Propagation
  Forward propagation is the process of passing input data through the network to generate predictions.
"""

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
# testing forward propagation
nn = NeuralNetwork(2, 3, 1)
X = np.array([0.5, 0.1])
output = nn.forward(X)
print("Network output:", output)