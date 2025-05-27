import matplotlib.pyplot as plt
from sklearn import datasets
from pca import PCA


# loading the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2 primary components
pca = PCA(2)

# fitting the model
pca.fit(X)

# transforming the data
X_transformed = pca.transform(X)

# printing the shape of the transformed data
print("Original shape of data:", X.shape)
print("Shape of transformed data:", X_transformed.shape)

# extracting the first two dimensions
x1 = X_transformed[:, 0]
x2 = X_transformed[:, 1]

# plotting the transformed data
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x1, x2, c=y, cmap='viridis', edgecolor='k', s=100)
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
cbar = plt.colorbar(scatter)
cbar.set_label('Target class (species)')
plt.show()