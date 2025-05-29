from svm import SVM
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# Generate synthetic data for binary classification
X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
y = np.where(y == 0, -1, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Train the SVM model
clf = SVM()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Evaluate the model's accuracy
def accuracy(y_true, y_pred):
  accuracy = np.sum(y_true == y_pred) / len(y_true)
  return accuracy

print("SVM classification accuracy", accuracy(y_test, predictions))

# Visualize the SVM decision boundary
def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Create meshgrid for background coloring
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    # Predict over the mesh using SVM decision function
    Z = np.array([clf.predict(np.array([[x, y]]))[0] for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    # Background color
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)

    # Plot the original points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=50, edgecolors='k')

    # Draw hyperplane and margins
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], color='blue', linestyle='--', linewidth=2, label='Hyperplane')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], color='red', linestyle='-', linewidth=1.5, label='Margin -1')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], color='green', linestyle='-', linewidth=1.5, label='Margin +1')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend()
    plt.grid(True)
    plt.title("SVM Decision Boundary with Background")
    plt.show()

visualize_svm()