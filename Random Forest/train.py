from sklearn import datasets
from random_forest import RandomForest
from sklearn.model_selection import train_test_split
import numpy as np


df = datasets.load_breast_cancer() # Load the breast cancer dataset from sklearn
X = df.data # Features of the dataset
y = df.target # Target variable of the dataset

# splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def accuracy(y_true, y_pred): # Function to calculate accuracy of predictions
  accuracy = np.sum(y_true == y_pred) / len(y_true) # Calculate accuracy as the ratio of correct predictions to total predictions
  return accuracy

# Create a Random Forest model with specified parameters
rf = RandomForest()
rf.fit(X_train, y_train) # Fit the Random Forest model to the training data
y_pred = rf.predict(X_test) # Make predictions on the test data
print("Accuracy:", accuracy(y_test, y_pred)) # Print the accuracy of the model on the test data