from threading import Thread

import matplotlib.pyplot as plt
import numpy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class DimensionalityReduction(Thread):
    def __init__(self, X, y):
        super(DimensionalityReduction, self).__init__()

        self.X = X
        self.y = y

        self.start()

    def run(self):
        self.plot_PCA()
        self.plot_LDA()

    def plot_PCA(self):
        pca = PCA(n_components=2)
        X_r = pca.fit_transform(self.X)

        # Percentage of variance explained for each components
        print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))

        plt.figure()
        for c, i, target_name in zip("rg", [0, 1], ["x", "y"]):
            # x_r = X_r[self.y == i, 0]
            # plt.scatter(X_r[self.y == i, 0], numpy.zeros(x_r.__len__()), c=c, label=target_name)

            plt.scatter(X_r[self.y == i, 0], X_r[self.y == i, 1], c=c, label=target_name)

        plt.legend()
        plt.title('PCA')

        plt.show()

    def plot_LDA(self):
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda = lda.fit(self.X, self.y)
        X_r2 = lda.transform(self.X)

        plt.figure()
        for c, i, target_name in zip("rg", [0, 1], ["x", "y"]):
            x_r2 = X_r2[self.y == i, 0]
            plt.scatter(x_r2, numpy.zeros(x_r2.__len__()), c=c, label=target_name)

        plt.legend()
        plt.title('LDA')
