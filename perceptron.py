import numpy as np

class Perceptron(object):

    def __init__(self, learning_rate=0.01, n_iter=10, random_state=1):
        #learning_rate : float -> learning rate
        #n_iter : int          -> epochs(passes over training set)
        #random_state : int    -> random number generator seed for random weight init
        #w_ : None             -> weight array -- initialized during fit
        #errors_ : list        -> list number of misclassifications in each epoch

        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None
        self.errors_ = []

    def predict(self, X):
        """return class label after unit step"""
        return np.where(self.net_input(X) >=  0.0, 1, -1)

    def net_input(self, X):
        """weighted sum of inputs"""
        # self.w_[0] == bias, which is why
        # self.w_[1:] == all weights except weight 0
        # w_[0] is the bias because it is eaiser to compute,
        # without having it be a separate var
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def fit(self, X, y) :
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) # X.shape[1] makes matrix with one feature

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.w_[1:] += xi * update
                self.w_[0] += update
                errors += int(update != 0.0)

            self.errors_.append(errors)

        return self



