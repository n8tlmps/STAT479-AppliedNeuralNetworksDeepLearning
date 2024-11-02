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

plt.figure(figsize=(12, 8))
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

"""
Loss Function
    The loss function measures how well our network is performing. We'll use the mean squared error(MSE) loss function
"""

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# test the MSE loss function
y_true = np.array([[1], [0], [1]])
y_pred = np.array([[0.9], [0.1], [0.8]])

loss = mse_loss(y_true, y_pred)
print("MSE Loss:", loss)

"""
Backpropagation
    Backpropagation is the algorithm used to calculate gradients and update weights.
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
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Output layer
        dZ2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer
        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# Initialize a neural network and perform one backward pass
nn = NeuralNetwork(2, 3, 1)
X = np.array([[0.5, 0.1]])
y = np.array([[1]])
nn.forward(X)
nn.backward(X, y , learning_rate=0.1)

"""
Training Loop
    Let's combine forward propagation, loss calculation, and back propagation into a training loop
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

"""
Loss Function
    The loss function measures how well our network is performing. We'll use the mean squared error(MSE) loss function
"""

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# test the MSE loss function
y_true = np.array([[1], [0], [1]])
y_pred = np.array([[0.9], [0.1], [0.8]])

loss = mse_loss(y_true, y_pred)
print("MSE Loss:", loss)

"""
Backpropagation
    Backpropagation is the algorithm used to calculate gradients and update weights.
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
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Output layer
        dZ2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer
        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # forward propagation
            output = self.forward(X)

            # compute loss
            loss = mse_loss(y, output)

            # backpropagation
            self.backward(X, y, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss}")

# training the NN on a simple dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([
    [0],
    [1],
    [1],
    [0]
]) # XOR function

nn = NeuralNetwork(2, 4, 1)
nn.train(X, y, epochs=1000, learning_rate=0.1)

"""
Making Predictions
    After training, we use our neural network to make predictions on new data.
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
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Output layer
        dZ2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer
        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # forward propagation
            output = self.forward(X)

            # compute loss
            loss = mse_loss(y, output)

            # backpropagation
            self.backward(X, y, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)
    
# making predictions using the trained network
nn = NeuralNetwork(2, 4, 1)
nn.train(X, y, epochs=5000, learning_rate=0.1)

test_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 0]
])
predictions = nn.predict(test_data)

for input_data, prediction in zip(test_data, predictions):
    print(f"Input: {input_data}, Prediction: {prediction[0]:.4f}")

"""
Visualizing the Decision Boundary
    We can visualize how our neural network separates the input space for the XOR problem
"""

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("XOR Decision Boundary")
    plt.show()

plot_decision_boundary(X, y, nn)

"""
Adding Regularization
    To prevent overfitting, we can add L2 regularization to our neural network
"""

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lambda_reg=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        self.lambda_reg = lambda_reg

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Output layer
        dZ2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer
        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Add L2 regularization terms
        dW2 += (self.lambda_reg / m) * self.W2
        dW1 += (self.lambda_reg / m) * self.W1

        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # forward propagation
            output = self.forward(X)

            # compute loss
            loss = mse_loss(y, output)

            # backpropagation
            self.backward(X, y, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)
    
# training the NN with regularization
nn_reg = NeuralNetwork(2, 4, 1, lambda_reg=0.1)
nn_reg.train(X, y, epochs=5000, learning_rate=0.1)

plot_decision_boundary(X, y, nn_reg)

"""
Mini-batch Gradient Descent
    To improve training efficiency, we can implement mini-batch gradient descent
"""

def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size

    for i in range(n_minibatches):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        y_mini = mini_batch[:, -1].reshape((-1, 1))

    return mini_batches

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lambda_reg=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        self.lambda_reg = lambda_reg

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Output layer
        dZ2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer
        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Add L2 regularization terms
        dW2 += (self.lambda_reg / m) * self.W2
        dW1 += (self.lambda_reg / m) * self.W1

        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # forward propagation
            output = self.forward(X)

            # compute loss
            loss = mse_loss(y, output)

            # backpropagation
            self.backward(X, y, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)

    def train(self, X, y, epochs, learning_rate, batch_size):
        for epoch in range(epochs):
            mini_batches = create_mini_batches(X, y, batch_size)

            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch
                self.forward(X_mini)
                self.backward(X_mini, y_mini, learning_rate)

            if epoch % 100 == 0:
                loss = mse_loss(y, self.predict(X))
                print(f"Epoch: {epoch}, Loss: {loss}")

nn_mini_batch = NeuralNetwork(2, 4, 1)
nn_mini_batch.train(X, y, epochs=5000, learning_rate=0.1, batch_size=2)

"""
Adding Momentum
    Momentum can help accelerate gradient descent and dampen oscillations.
"""

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lambda_reg=0.01, momentum=0.9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        self.lambda_reg = lambda_reg

        self.momentum = momentum
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Output layer
        dZ2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer
        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Add L2 regularization terms
        dW2 += (self.lambda_reg / m) * self.W2
        dW1 += (self.lambda_reg / m) * self.W1

        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        # Update velocities
        self.v_W2 = self.momentum * self.v_W2 + learning_rate * dW2
        self.v_b2 = self.momentum * self.v_b2 + learning_rate * db2
        self.v_W1 = self.momentum * self.v_W1 + learning_rate * dW1
        self.v_b1 = self.momentum * self.v_b1 + learning_rate * db1

        # Update weights and biases
        self.W2 -= self.v_W2
        self.b2 -= self.v_b2
        self.W1 -= self.v_W1
        self.b1 -= self.v_b1

    # def train(self, X, y, epochs, learning_rate):
    #     for epoch in range(epochs):
    #         # forward propagation
    #         output = self.forward(X)

    #         # compute loss
    #         loss = mse_loss(y, output)

    #         # backpropagation
    #         self.backward(X, y, learning_rate)

    #         if epoch % 100 == 0:
    #             print(f"Epoch: {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)

    def train(self, X, y, epochs, learning_rate, batch_size):
        for epoch in range(epochs):
            mini_batches = create_mini_batches(X, y, batch_size)

            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch
                self.forward(X_mini)
                self.backward(X_mini, y_mini, learning_rate)

            if epoch % 100 == 0:
                loss = mse_loss(y, self.predict(X))
                print(f"Epoch: {epoch}, Loss: {loss}")

# train the NN with momentum
nn_momentum = NeuralNetwork(2, 4, 1, momentum=0.9)
nn_momentum.train(X, y, epochs=5000, learning_rate=0.1, batch_size=2)