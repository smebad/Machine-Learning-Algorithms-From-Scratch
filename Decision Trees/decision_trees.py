# Decision Trees from scratch
import numpy as np
from collections import Counter

class Node: # Node class for decision tree
  def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None): # Constructor, p.s we put '*' to allow value to be optional
    self.feature = feature # Feature index to split on
    self.threshold = threshold # Threshold value to split on
    self.left = left # Left child node
    self.right = right # Right child node
    self.value = value # Value for leaf node (class label or regression value)

  def is_leaf(self): # Check if the node is a leaf node
    return self.value is not None


class DecisionTree: # Decision tree class
  def __init__(self, min_samples_split=2, max_depth=100, n_features=None): # Constructor with parameters for minimum samples to split and maximum depth
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth
    self.n_features = n_features
    self.root = None


  def fit(self, X, y): # Fit the decision tree to the training data
    self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features) # Number of features to consider for splits
    self.root = self._grow_tree(X, y, depth=0) # Recursive function to grow the tree


  def _grow_tree(self, X, y, depth = 0):
    n_samples, n_feats = X.shape
    n_labels = len(np.unique(y))
    
    # checking the stopping criteria
    if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
      # If stopping criteria met, return a leaf node with the most common label
      leaf_value = self._most_common_label(y)
      return Node(value = leaf_value)
    
    # Randomly select features for splitting if n_features is specified
    features_idx = np.random.choice(n_feats, self.n_features, replace=False) 
      

    # finding the best split
    best_feature, best_threshold = self._best_split(X, y, features_idx)

    # creating the child nodes
    left_idx, right_idx = self._split(X[:, best_feature], best_threshold) # Split the data into left and right child nodes
    left_child = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1) # Recursively grow the left child node
    right_child = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1) # Recursively grow the right child node

    return Node(best_feature, best_threshold, left_child, right_child)

  def _best_split(self, X, y, features_idx):
    best_gain = -1
    split_idx, split_threshold = None, None

    # Iterate over each feature index to find the best split
    for feature_idx in features_idx:
      X_column = X[:, feature_idx]
      thresholds = np.unique(X_column) # Get unique values in the feature column

      # Iterate over each threshold to find the best gain
      for threshold in thresholds:
        # calculate the information gain for the split
        gain = self._information_gain(y, X_column, threshold)

        if gain > best_gain:
          best_gain = gain
          split_idx = feature_idx
          split_threshold = threshold
    return split_idx, split_threshold
  

  def _information_gain(self, y, X_column, threshold):
    # parent entropy for the current node
    parent_entropy = self._entropy(y)

    # creating child nodes based on the threshold
    left_idx, right_idx = self._split(X_column, threshold) # Split the data into left and right child nodes

    # If either left or right child is empty, return 0 gain
    if len(left_idx) == 0 or len(right_idx) == 0:
      return 0    
    
    # calculating the weighted entropy of the child nodes
    n = len(y) # Total number of samples
    n_left, n_right = len(left_idx), len(right_idx)
    e_left, e_right = self._entropy(y[left_idx]), self._entropy(y[right_idx]) # Entropy of left and right child nodes
    child_entropy = (n_left / n) * e_left + (n_right / n) * e_right # Weighted average of child entropies


    #  calculating the information gain
    information_gain = parent_entropy - child_entropy # Information gain is the reduction in entropy after the split
    return information_gain
  

  def _split(self, X_column, split_threshold):
    left_idx = np.argwhere(X_column <= split_threshold).flatten()  # Indices where the feature value is less than or equal to the threshold
    right_idx = np.argwhere(X_column > split_threshold).flatten()  # Indices where the feature value is greater than the threshold
    return left_idx, right_idx


  def _entropy(self, y):
    hist = np.bincount(y) # Count occurrences of each label
    ps = hist / len(y)  # Normalize the histogram to get probabilities
    return -np.sum([p * np.log2(p) for p in ps if p > 0]) # Calculate entropy using the formula -Σ(p * log2(p))

  def _most_common_label(self, y):
    counter = Counter(y) # Count occurrences of each label
    value = counter.most_common(1)[0][0] # Get the most common label
    return value


  def predict(self, X): # Predict class labels for input data
    return np.array([self._traverse_tree(x, self.root) for x in X]) # Traverse the tree for each input data point and return the predicted labels

  def _traverse_tree(self, x, node): # Traverse the tree to find the predicted label for a single input data point 
    if node.is_leaf():
      return node.value
    
    if x[node.feature] <= node.threshold:
      return self._traverse_tree(x, node.left)
    return self._traverse_tree(x, node.right) # Traverse to the right child if the feature value is greater than the threshold