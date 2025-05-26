# Naive Bayes Classifier Implementation from scratch
import numpy as np


class NaiveBayes:

  def fit(self, X, y):
    n_samples, n_features = X.shape
    self._classes = np.unique(y)
    n_classes = len(self._classes)

    # calculating the mean, variance and prior for each class
    self._mean = np.zeros((n_classes, n_features), dtype = np.float64) # mean for each class
    self._variance = np.zeros((n_classes, n_features), dtype = np.float64) # variance for each class
    self._priors = np.zeros(n_classes, dtype = np.float64) # prior for each class

    for idx, c in enumerate(self._classes): # iterating through each class
      X_c = X[y == c] # selecting the samples for class c
      self._mean[idx, :] = X_c.mean(axis = 0) # calculating the mean for class c
      self._variance[idx, :] = X_c.var(axis=0) # calculating the variance for class c
      self._priors[idx] = X_c.shape[0] / float(n_samples) # calculating the prior for class c


  def predict(self, X):
    y_pred = [self._predict(x) for x in X] # predicting the class for each sample in X
    return np.array(y_pred)
  
  def _predict(self, x):
    posteriors = []

    # calculating the posteriors probability for each class
    for idx, c in enumerate(self._classes): # iterating through each class
      prior = np.log(self._priors[idx]) # calculating the prior probability for class c
      posterior = np.sum(np.log(self._pdf(idx, x)))
      posterior = posterior + prior # adding the prior to the posterior
      posteriors.append(posterior) # appending the posterior to the list

    # return the class with the highest posterior
    return self._classes[np.argmax(posteriors)]
  
  def _pdf(self, class_idx, x): # calculating the probability density function for class class_idx
    mean = self._mean[class_idx]
    variance = self._variance[class_idx]
    numerator = np.exp(-((x - mean) ** 2) / (2 * variance)) # calculating the numerator of the PDF
    denominator = np.sqrt(2 * np.pi * variance) # calculating the denominator of the PDF
    return numerator / denominator # returning the PDF value for class class_idx and sample x
  