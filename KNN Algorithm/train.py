import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN


cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()


clf = KNN(k = 3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(predictions)

# testing accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f'Accuracy: {accuracy:.2f}')
