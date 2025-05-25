# Random Forest from scratch
import numpy as np
from collections import Counter
from decision_trees_for_the_class import DecisionTree



class RandomForest:
  
  def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
    self.n_trees = n_trees
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.n_features = n_features
    self.trees = []

  # Fit the Random Forest model to the training data
  def fit(self, X, y):
    self.trees = []

    # for loop to create multiple trees
    for _ in range(self.n_trees):
      tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=self.n_features)

      X_sample, y_sample = self._bootstrap_sample(X, y)
      tree.fit(X_sample, y_sample)
      self.trees.append(tree)


  # Bootstrap sampling to create a random subset of the training data
  def _bootstrap_sample(self, X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return X[indices], y[indices]
  
  # Get the most common label from the predictions of the trees
  def _most_common_label(self, y):
    counter = Counter(y) # Count occurrences of each label
    most_common = counter.most_common(1)[0][0] # Get the most common label
    return most_common

  # Make predictions on the test data
  def predict(self, X):
    predictions = np.array([tree.predict(X) for tree in self.trees])
    tree_predictions = np.swapaxes(predictions, 0, 1)
    predictions = np.array([self._most_common_label(pred) for pred in tree_predictions])
    return predictions