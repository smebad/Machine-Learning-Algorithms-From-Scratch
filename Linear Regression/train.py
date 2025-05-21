import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from Linear_regression import LinearRegression


# Loading the dataset
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=4)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# plotting the dataset
fig = plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y, color='red', label='Data points', marker='o', s=30)
plt.title('Dataset')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Creating an instance of the LinearRegression class
reg = LinearRegression(lr = 0.01)

# Fitting the model to the training data
reg.fit(X_train, y_train)

# Making predictions on the test data
predictions = reg.predict(X_test)

# Calculating the mean squared error
def mse(y_test, predictions):
  return np.mean(y_test - predictions) ** 2 # mean squared error


mse = mse(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Predict line values
y_pred_line = reg.predict(X)

# Sort X for proper line plotting
sorted_indices = X[:, 0].argsort()
X_sorted = X[sorted_indices]
y_pred_sorted = y_pred_line[sorted_indices]

# Plotting the predictions
y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(10, 6))
n1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10, label='Training data')
n2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10, label='Testing data')
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Regression line')
plt.title('Linear Regression')
plt.show()