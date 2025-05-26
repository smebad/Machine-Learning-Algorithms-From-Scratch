# Naive Bayes Classifier from Scratch

## What is Naive Bayes in Machine Learning?

Naive Bayes is a family of simple probabilistic classifiers based on applying Bayes' Theorem with strong (naive) independence assumptions between features. It is primarily used for classification tasks and is known for its efficiency and effectiveness on large datasets. It assumes that the presence of a particular feature in a class is independent of the presence of other features.

Bayes' Theorem:

$$
P(C|X) = \frac{P(X|C) * P(C)}{P(X)}
$$

Where:

* $P(C|X)$ is the posterior probability of class C given predictor X.
* $P(X|C)$ is the likelihood of predictor X given class C.
* $P(C)$ is the prior probability of class C.
* $P(X)$ is the prior probability of predictor X.

## Project Overview

This project contains two Python files:

* `naive_bayes.py`: Contains the implementation of the Naive Bayes classifier from scratch using NumPy.
* `train.py`: Tests the classifier using a synthetic dataset from Scikit-learn.

## Step-by-Step Code Explanation

### 1. `naive_bayes.py`

This file defines a `NaiveBayes` class.

#### `fit()` method:

* Computes statistics for each class:

  * **Mean**: Average of feature values per class.
  * **Variance**: Variance of feature values per class.
  * **Prior**: Probability of each class occurring in the training data.

#### `predict()` method:

* Predicts the class for each input sample by calling the `_predict()` method.

#### `_predict()` method:

* Calculates the **posterior probability** for each class:

  * Applies log of the prior.
  * Adds the log of the **probability density function (PDF)** for each feature.
  * Selects the class with the highest posterior probability.

#### `_pdf()` method:

* Computes the **Gaussian probability density** for a feature:

  * Formula: $\frac{e^{-\frac{(x - \mu)^2}{2\sigma^2}}}{\sqrt{2\pi\sigma^2}}$

### 2. `train.py`

This script tests the model:

* Generates a synthetic binary classification dataset using `make_classification()`.
* Splits the data into training and testing sets using `train_test_split()`.
* Trains the `NaiveBayes` model.
* Predicts on the test set.
* Calculates and prints the accuracy.

## Prediction and Accuracy Score

After training and testing the model, the accuracy score obtained was:

**Accuracy: 0.81**

This means that 81% of the test samples were correctly classified. This is a good result for a basic implementation and shows the model is working effectively.

## What I Learned

* How Bayes' Theorem is applied in classification.
* How to compute mean, variance, and prior probabilities from scratch.
* How to compute Gaussian probability for continuous features.
* How to use NumPy for efficient mathematical operations.
* How to evaluate model performance using accuracy.
* How models work internally beyond Scikit-learn's abstractions.

Implementing Naive Bayes from scratch helped deepen my understanding of probabilistic classifiers and how statistical assumptions play a role in machine learning. It also gave me hands on experience with vectorized computations and writing reusable code.

---
