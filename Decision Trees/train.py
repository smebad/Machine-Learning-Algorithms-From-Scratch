from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from decision_trees import DecisionTree

df = datasets.load_breast_cancer()
X = df.data
y = df.target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = DecisionTree()
clf.fit(X_train, y_train) # Fitting the decision tree to the training data

predictions = clf.predict(X_test) # Making predictions on the test set

# Evaluating the accuracy of the model
def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test) * 100 # Calculating accuracy as a percentage

# accuracy
acc = accuracy(y_test, predictions)
print(f"Accuracy: {acc:.2f}%") # Printing the accuracy of the model
