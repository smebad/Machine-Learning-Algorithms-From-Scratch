# Random Forest from Scratch

This project implements the **Random Forest** algorithm from scratch using Python and NumPy, without relying on any machine learning libraries for model construction. It includes a custom-built Decision Tree class and combines multiple trees to build a Random Forest classifier. The model is tested using the Breast Cancer dataset from scikit-learn.

---

## What is Random Forest?

**Random Forest** is an ensemble learning algorithm used for classification and regression. It builds multiple decision trees during training and outputs the class that is the mode (majority vote) of the classes predicted by individual trees. Key concepts include:

* **Bagging (Bootstrap Aggregating)**: Each tree is trained on a different random subset of the training data.
* **Random Feature Selection**: Each split in each tree considers a random subset of features.
* **Majority Voting**: Final predictions are made by taking the majority vote across all trees.

Random Forests are powerful because they reduce overfitting, improve accuracy, and handle high-dimensional data well.

---

## Files and Their Purpose

### 1. `decision_trees_for_the_class.py`

Contains the custom `DecisionTree` class used for building individual decision trees. Key components include:

* Recursive tree-building logic
* Entropy and information gain calculation
* Splitting based on best features
* Prediction by tree traversal

### 2. `random_forest.py`

Implements the `RandomForest` class. Key parts:

* `fit()`: Trains multiple decision trees on random subsets of the data using bootstrapping.
* `_bootstrap_sample()`: Generates a new sample of data for each tree.
* `_most_common_label()`: Combines predictions from all trees using majority vote.
* `predict()`: Makes predictions for test samples by aggregating predictions from each tree.

### 3. `train.py`

This script:

* Loads the Breast Cancer dataset
* Splits the data into training and test sets
* Trains the `RandomForest` model
* Evaluates the model using accuracy
* Prints the final accuracy score

---

## Code Explanation (with Comments)

### `random_forest.py`

* `__init__`: Initializes the forest with a given number of trees, tree depth, sample size, and number of features to consider.
* `fit()`: Trains `n_trees` decision trees, each on a random bootstrap sample of the training data.
* `_bootstrap_sample()`: Randomly samples with replacement to create diverse training sets.
* `predict()`: Gets predictions from all trees and returns the most common prediction for each sample.

### `train.py`

* Uses `sklearn.datasets` to load Breast Cancer data.
* Splits the dataset using `train_test_split()`.
* Trains the `RandomForest` model using the `fit()` method.
* Predicts labels using `predict()`.
* Evaluates accuracy using a custom `accuracy()` function.

---

## Prediction and Accuracy Score

The model makes predictions by aggregating outputs from all decision trees in the forest. The most common label among the trees is chosen for each sample:

```python
predictions = np.array([tree.predict(X) for tree in self.trees])
tree_predictions = np.swapaxes(predictions, 0, 1)
predictions = np.array([self._most_common_label(pred) for pred in tree_predictions])
```

The accuracy is calculated as:

```python
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
```

### Final Accuracy:

```
Accuracy: 0.9736842105263158
```

This means the model correctly predicted about **97.36%** of the samples in the test set.

---

## Improving the Model (Manual Hyperparameter Tuning)

To improve the model, you can manually tune these parameters in the `RandomForest` constructor:

```python
rf = RandomForest(n_trees=20, max_depth=15, min_samples_split=4, n_features=8)
```

* **n\_trees**: More trees generally improve accuracy but increase computation time.
* **max\_depth**: Prevents trees from growing too deep and overfitting.
* **min\_samples\_split**: Avoids splitting nodes that don't have enough samples.
* **n\_features**: Controls feature randomness. Smaller values increase tree diversity.

Use trial and error or grid search to find the best combination.

---

## What I Learned

Implementing Random Forest from scratch helped me:

* Deeply understand how decision trees work and how they are used in ensembles.
* Learn how bootstrap sampling and feature randomness contribute to model robustness.
* Gain practical experience in recursive tree building, prediction aggregation, and error analysis.
* Appreciate the power of ensemble methods in reducing overfitting and improving performance.

This project has enhanced my understanding of both the theory and coding of one of the most popular machine learning algorithms in use today.

---