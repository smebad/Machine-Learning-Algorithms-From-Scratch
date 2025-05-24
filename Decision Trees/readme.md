# Decision Trees from Scratch

## What is a Decision Tree?

A Decision Tree is a supervised machine learning algorithm that is commonly used for classification and regression tasks. It works by splitting the data into subsets based on the value of input features. Each node in the tree represents a decision based on a feature, and each leaf node represents the final output or prediction. The process of building a tree involves selecting the best feature and threshold that yields the highest information gain (i.e., the most reduction in impurity or entropy).

## What This Project Does

This project implements a Decision Tree algorithm from scratch using Python and NumPy, without relying on any machine learning libraries. It consists of two main files:

* `decision_trees.py`: Contains the implementation of the `DecisionTree` and `Node` classes, which together define the structure and behavior of the decision tree.
* `train.py`: Loads the dataset, splits it into training and testing sets, trains the decision tree model, makes predictions, and evaluates the model's accuracy.

## Code Explanation with Comments

### decision\_trees.py

* **Node class**: Represents a single node in the tree. It contains information like which feature and threshold were used to split, references to child nodes, or the final value if it is a leaf node.
* **DecisionTree class**: Handles the training (`fit`), prediction (`predict`), and splitting logic.

  * `_grow_tree`: Builds the tree recursively by splitting on the best feature and threshold.
  * `_best_split`: Finds the feature and threshold that result in the best information gain.
  * `_information_gain`: Measures how much a split improves the purity of the nodes.
  * `_split`: Performs the actual data split into left and right subsets.
  * `_entropy`: Calculates the entropy of a label distribution.
  * `_most_common_label`: Determines the most frequent class in a subset.
  * `_traverse_tree`: Traverses the tree from root to leaf to make a prediction for a single data point.

### train.py

* Loads the Breast Cancer dataset using scikit-learn.
* Splits the data into training and testing sets.
* Trains the decision tree using the `fit` method.
* Makes predictions on the test set.
* Calculates and prints the accuracy using a custom `accuracy` function.

## Prediction and Accuracy

The model predicts whether a tumor is malignant or benign. The prediction is done by traversing the decision tree for each test data point. The accuracy is calculated by comparing the predicted values against the actual values in the test set.

Example output:

```
Accuracy: 93.86%
```

This means the model correctly predicted the labels for approximately 93.86% of the test samples.

## Improvement Using `max_depth`

Initially, the decision tree might overfit if allowed to grow indefinitely. To address this, a `max_depth` parameter is used to control the maximum depth of the tree.

Limiting the depth helps prevent overfitting by stopping the tree from modeling noise in the data. This results in better generalization to unseen data.

Example usage:

```python
clf = DecisionTree(max_depth=10)
```

This restricts the tree to a maximum of 10 levels deep.

## What I Learned

Implementing Decision Trees from scratch taught me:

* How decision trees work under the hood.
* How to compute entropy and information gain.
* How tree structures are recursively built and traversed.
* The importance of hyperparameters like `max_depth` in controlling model complexity.
* A deeper understanding of classification tasks and how machine learning models learn from data.

This project was a valuable experience in understanding core machine learning concepts without relying on high-level libraries.
