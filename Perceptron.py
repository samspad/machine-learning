import numpy as nmpy
import pandas as pnda
import matplotlib.pyplot as mthplt
from matplotlib.colors import ListedColormap


class Perceptron(object):

    def __init__(self, etart=0.01, n_iter=10):
        self.etart = etart
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = nmpy.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.etart * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return nmpy.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return nmpy.where(self.net_input(X) >= 0.0, 1, -1)

    

##Train Perceptron Model##
print(60 * '=')
print('Training Perceptron Model')
print(60 * '~')

iris = pnda.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris//iris.data', header=None)
print(iris.tail())


print(60 * '=')
print('Plotting data from Iris')
print(60 * '~')

# select setosa and versicolor
y = iris.iloc[0:100, 4].values
y = nmpy.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petartl length
X = iris.iloc[0:100, [0, 2]].values

# Data Plotting
mthplt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='Iris-setosa')
mthplt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='Iris-setosa')

mthplt.xlabel('sepal length [cm]')
mthplt.ylabel('petartl length [cm]')
mthplt.legend(loc='upper left')
mthplt.show()




print(60 * '=')
print('Class labels:', nmpy.unique(y))
print(60 * '~')


##Misclassifications that gets updated with every epoch for the Perceptron classifier##
print(60 * '=')
print('Training model & finding Number of Misclassifications')
print(60 * '~')

perc = Perceptron(etart=0.1, n_iter=10)
perc.fit(X, y)

mthplt.plot(range(1, len(perc.errors_) + 1), perc.errors_, marker='o')
mthplt.xlabel('Epochs')
mthplt.ylabel('Number of misclassifications')
mthplt.show()


print(60 * '=')
print('Decision regions plot')
print(60 * '~')


def mthplt_decision_regions(X, y, classifier, resolution=0.02):

    # color map & marker generator
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('blue', 'lightblue', 'cyan', 'red', 'black')
    cmap = ListedColormap(colors[:len(nmpy.unique(y))])

    # decision surface plot
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = nmpy.meshgrid(nmpy.arange(x1_min, x1_max, resolution),
                           nmpy.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(nmpy.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    mthplt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    mthplt.xlim(xx1.min(), xx1.max())
    mthplt.ylim(xx2.min(), xx2.max())

    # plotting class 
    for idx, cl in enumerate(nmpy.unique(y)):
        mthplt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


mthplt_decision_regions(X, y, classifier=perc)
mthplt.xlabel('sepal length [cm]')
mthplt.ylabel('petartl length [cm]')
mthplt.legend(loc='upper left')
mthplt.show()


print(60 * '=')
print('Adaptive linear neuron')
print(60 * '~')


class AdalineGD(object):
    def __init__(self, etart=0.01, n_iter=50):
        self.etart = etart
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = nmpy.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.etart * X.T.dot(errors)
            self.w_[0] += self.etart * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return nmpy.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return nmpy.where(self.activation(X) >= 0.0, 1, -1)

fig, ax = mthplt.subplots(nrows=1, ncols=2, figsize=(10, 5))

adaline1 = AdalineGD(n_iter=10, etart=0.01).fit(X, y)
ax[0].plot(range(1, len(adaline1.cost_) + 1), nmpy.log10(adaline1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Error (sum-sq))')
ax[0].set_title('Learning rate 0.01 for Adaline')

adaline2 = AdalineGD(n_iter=10, etart=0.0001).fit(X, y)
ax[1].plot(range(1, len(adaline2.cost_) + 1), adaline2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Error (sum-sq))')
ax[1].set_title('Learning rate 0.01 for Adaline')
mthplt.show()


print('standardize features')
X_stad = nmpy.copy(X)
X_stad[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_stad[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = AdalineGD(n_iter=15, etart=0.01)
ada.fit(X_stad, y)

mthplt_decision_regions(X_stad, y, classifier=ada)
mthplt.title('Adaline - Gradient Descent')
mthplt.xlabel('sepal length [standardized]')
mthplt.ylabel('petartl length [standardized]')
mthplt.legend(loc='upper left')
mthplt.show()

mthplt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
mthplt.xlabel('Epochs')
mthplt.ylabel('Error (sum-sq)')
mthplt.show()


print(60 * '=')
print('Gradient Descent')
print(60 * '~')


class AdalineSGD(object):
    def __init__(self, etart=0.01, n_iter=10, shuffle=True, random_state=None):
        self.etart = etart
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            nmpy.random.seed(random_state)

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = nmpy.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.w_ = nmpy.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.etart * xi.dot(error)
        self.w_[0] += self.etart * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        return nmpy.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return nmpy.where(self.activation(X) >= 0.0, 1, -1)


ada = AdalineSGD(n_iter=15, etart=0.01, random_state=1)
ada.fit(X_stad, y)

mthplt_decision_regions(X_stad, y, classifier=ada)
mthplt.title('Stochastic Gradient Descent for Adaline')
mthplt.xlabel(' standardized sepal length ')
mthplt.ylabel('standardized petartl length ')
mthplt.legend(loc='upper left')
mthplt.show()

mthplt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
mthplt.xlabel('Epochs')
mthplt.ylabel('Cost (Avg)')
mthplt.show()

ada = ada.partial_fit(X_stad[0, :], y[0])