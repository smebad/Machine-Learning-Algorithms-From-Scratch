import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from naive_bayes import NaiveBayes  # Import your class here

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Generate synthetic binary classification data
X, y = datasets.make_classification(
    n_samples=1000, n_features=10, n_classes=2, random_state=42
)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and test your Naive Bayes model
nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print("Accuracy:", accuracy(y_test, predictions))
