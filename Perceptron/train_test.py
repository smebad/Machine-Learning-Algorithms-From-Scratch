import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from perceptron import Perceptron


# Load dataset
X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.05) # Generate synthetic dataset with two classes

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Perceptron
perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
# Train Perceptron
perceptron.fit(X_train, y_train)

# accuracy function
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true) # Calculate accuracy as the ratio of correct predictions to total predictions
    return accuracy

# Test Perceptron
y_pred = perceptron.predict(X_test)

# Calculate accuracy
acc = accuracy(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Plot decision boundary
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

plot_decision_boundary(X, y, perceptron)