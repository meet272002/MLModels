import numpy as np

class PCA_Eigen:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X):
        self.mean = np.mean(X,axis=0)
        X_centered = X - self.mean

        cov = np.cov(X_centered, rowvar=False)

        eigenvalues,eigenvectors = np.linalg.eigh(cov)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:,sorted_idx]

        if self.n_components is not None:
            eigenvectors = eigenvectors[:,:self.n_components]
            eigenvalues = eigenvalues[:self.n_components]

        self.components = eigenvectors
        self.explained_variance = eigenvalues

    def transform(self,X):
        X_centered = X - self.mean
        return np.dot(X_centered,self.n_components)

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