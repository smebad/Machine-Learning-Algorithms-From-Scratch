# K-Nearest Neighbors (KNN) from Scratch

## Introduction
This project is an implementation of the K-Nearest Neighbors (KNN) algorithm from scratch using Python and NumPy. The goal was to understand the inner workings of KNN by building it manually instead of using existing libraries like scikit-learn. The project consists of two files:

- `KNN.py`: Implements the KNN algorithm.
- `train.py`: Trains and evaluates the KNN model using the Iris dataset.

## Implementation Details
### `KNN.py`
This file defines the `KNN` class and includes:
- **Euclidean Distance Function**: Computes the distance between two data points.
- **KNN Class**:
  - `__init__(self, k=3)`: Initializes the classifier with `k` nearest neighbors.
  - `fit(self, X, y)`: Stores the training data.
  - `predict(self, X)`: Predicts the class labels for given test data.
  - `_predict(self, x)`: Computes distances to all training samples, selects `k` nearest neighbors, and determines the most common class label.

### `train.py`
This file:
- Loads the **Iris dataset** from `sklearn.datasets`.
- Splits the data into training and testing sets using `train_test_split()`.
- Visualizes the dataset using `matplotlib`.
- Trains the `KNN` classifier using `clf.fit(X_train, y_train)`.
- Predicts test data labels and computes accuracy.

## What I Learned
1. **Understanding KNN**: Implementing KNN manually gave me a deeper understanding of how it works, especially how distance calculation and majority voting determine class labels.
2. **Euclidean Distance**: Learned how Euclidean distance is used to find nearest neighbors.
3. **Sorting and Majority Voting**: Gained insights into using `np.argsort()` for sorting distances and `Counter.most_common()` for selecting the majority class.
4. **Model Evaluation**: Understood how to evaluate model performance by calculating accuracy manually.
5. **Visualization**: Learned how to visualize data distribution to get insights into feature relationships.


This will train the model and output predictions along with accuracy.

