# Perceptron from scratch
import numpy as np

def unit_step_function(x):
    """Unit step activation function."""
    return np.where(x >= 0, 1, 0)
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.activation_function = unit_step_function # Unit step function
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape # Number of samples and features from the input data

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, 0) # Convert y to binary labels

        for _ in range(self.n_iterations): # Iterate over the number of iterations
            for idx, X_i in enumerate(X): # Iterate over each sample
                linear_output = np.dot(X_i, self.weights) + self.bias # Calculate linear output
                y_predicted = self.activation_function(linear_output) # Apply activation function

                # Update weights and bias
                update = self.lr * (y_[idx] - y_predicted) # Calculate update
                self.weights += update * X_i # Update weights
                self.bias += update # Update bias

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias # Calculate linear output
        y_predicted = self.activation_function(linear_output) # Apply activation function
        return y_predicted