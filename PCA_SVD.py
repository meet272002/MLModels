import numpy as np

# Vt -> signifies the eigen vector along the PCA direction.
# V is a principal direction in the feature space.
# U -> data points expressed in the new PCA coordinate system.
class PCA_Eigen:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.singular_values = None
        self.explained_variance = None

    def fit(self, X):
        self.mean = np.mean(X,axis=0)
        X_centered = X - self.mean

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        if self.n_components is None:
            self.components = Vt
        else:
            self.components = Vt[:self.n_components]

        n_samples = X.shape[0]
        self.singular_values = S
        self.explained_variance = (S**2) / (n_samples - 1)

    def transform(self,X):
        X_centered = X - self.mean
        return np.dot(X_centered,self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])

pca = PCA_Eigen(n_components=1)
X_reduced = pca.fit_transform(X)

print("Principal Component(s):\n", pca.components)
print("Explained Variance:", pca.explained_variance)
print("Reduced Data:\n", X_reduced)