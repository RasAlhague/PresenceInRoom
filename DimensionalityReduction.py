from multiprocessing import Process

import numpy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class DimensionalityReduction:
    def __init__(self, X, y, pca=True, lda=True):
        self.X = X
        self.y = y
        self.pca = pca
        self.lda = lda

        if pca:
            Process(target=self.plot_PCA).start()
        if lda:
            Process(target=self.plot_LDA).start()

    def plot_PCA(self):
        import matplotlib.pyplot as plt

        pca = PCA(n_components=5)
        X_r = pca.fit_transform(self.X)

        # Percentage of variance explained for each components
        print('explained variance ratio for pca: %s' % str(pca.explained_variance_ratio_))

        plt.figure()
        for c, i, target_name in zip("rgb", [0, 1, 2], ["1", "2", "3"]):
            # x_r = X_r[self.y == i, 0]
            # plt.scatter(X_r[self.y == i, 0], numpy.zeros(x_r.__len__()), c=c, label=target_name)

            plt.scatter(X_r[self.y == i, 0], X_r[self.y == i, 1], c=c, label=target_name)

        plt.legend()
        plt.title('PCA')
        plt.show()

    def plot_LDA(self):
        import matplotlib.pyplot as plt

        lda = LinearDiscriminantAnalysis(n_components=1)
        X_r2 = lda.fit_transform(self.X, self.y)

        plt.figure()
        for c, i, target_name in zip("rg", [0, 1], ["x", "y"]):
            x_r2 = X_r2[self.y == i, 0]
            plt.scatter(x_r2, numpy.zeros(x_r2.__len__()), c=c, label=target_name)

        plt.legend()
        plt.title('LDA')
        plt.show()
