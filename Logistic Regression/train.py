import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

# Load the dataset
df = datasets.load_breast_cancer()

# Split the dataset into training and testing sets
X, y = df.data, df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)


# Train the model
# Logistic Regression Classifier
classifier = LogisticRegression(lr = 0.01)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# accuracy function
def accuracy(y_pred, y_test):
  return np.sum(y_pred == y_test) / len(y_test)

# Calculate the accuracy
acc = accuracy(y_pred, y_test)
print(acc)