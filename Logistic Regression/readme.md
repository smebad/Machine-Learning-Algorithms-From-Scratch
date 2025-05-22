# Logistic Regression From Scratch

## What is Logistic Regression?

Logistic Regression is a type of supervised machine learning algorithm used for classification problems. Unlike linear regression which predicts continuous values, logistic regression predicts probabilities and classifies data points into discrete categories, typically 0 or 1. It uses the **sigmoid function** to map any real-valued number into a value between 0 and 1, making it suitable for binary classification tasks.

## What I Did in This Project

This project consists of two Python files:

* `logistic_regression.py`: Implements logistic regression from scratch using NumPy.
* `train.py`: Loads the breast cancer dataset, trains the model using the custom logistic regression class, and evaluates its performance.

---

## Code Explanation with Comments

### File: `logistic_regression.py`

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Activation function that maps values between 0 and 1

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr  # Learning rate
        self.n_iters = n_iters  # Number of iterations for gradient descent
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights
        self.bias = 0  # Initialize bias

        for _ in range(self.n_iters):  # Gradient descent loop
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_model)
        return [0 if y <= 0.5 else 1 for y in y_pred]  # Threshold at 0.5
```

### File: `train.py`

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from logistic_regression import LogisticRegression

# Load the breast cancer dataset
X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Train the model
classifier = LogisticRegression(lr=0.01)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Accuracy function
def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

# Evaluate the model
acc = accuracy(y_pred, y_test)
print(acc)  # Prints accuracy score
```

---

## Explanation of Prediction

The model makes predictions using the sigmoid function, which outputs probabilities. These probabilities are then thresholded at 0.5 to classify the result as either 0 or 1. For example, if the sigmoid output is 0.7, the prediction is class 1. If it is 0.3, the prediction is class 0.

---

## Accuracy, Error, and Learning Rate

In this project, the model achieved an accuracy of approximately `0.921`, or 92.1% on the test set.

* **Accuracy** is calculated by comparing predicted labels to the actual labels.
* **Error** is implicitly handled during training by minimizing the loss (via gradient descent).
* **Learning Rate** (`lr`) controls how big the steps are during optimization:

  * A very small learning rate can lead to slow training.
  * A very high learning rate can overshoot the optimal solution.

By setting the learning rate to `0.01`, I was able to improve training speed and achieve better convergence and accuracy.

---

## What I Learned From This Project

* How logistic regression works mathematically and programmatically.
* Implementing the sigmoid function and using it for binary classification.
* How gradient descent updates weights and bias in each iteration.
* How to evaluate model performance using accuracy.
* The effect of learning rate on model training and prediction performance.
* Hands-on practice of building machine learning models from scratch, without using libraries like Scikit-learn for modeling.

This project helped solidify my understanding of the core principles of machine learning and gave me confidence to build more models independently.