import arff
import numpy as np

class DataSet(object):
    def __init__(self, config):
        self.config = config
        self.train, self.test, self.validation = None, None, None
        self.path = self.config.dataset_path

    def get_data(self, path, noise=False):
        data = np.load(path)
        if noise == True :
            data = data + np.random.normal(0, 0.001, data.shape)
        return data

    def get_train(self):
        if self.train == None:
            X = self.get_data(self.config.train_path + "-features.pkl", True)
            Y = self.get_data(self.config.train_path + "-labels.pkl")
            length = X.shape[0]
            X, Y = X[0 : int(0.833 * length) , :], Y[0 : int(0.833 * length), :]
            self.train = X, Y
        else :
            X, Y = self.train

        X, Y = self.eliminate_data(X, Y)
        return X, Y

    def get_validation(self):
        if self.validation == None:
            X = self.get_data(self.config.train_path + "-features.pkl")
            Y = self.get_data(self.config.train_path + "-labels.pkl")
            length = X.shape[0]
            X, Y = X[0 : int(0.167 * length) , :], Y[0 : int(0.167 * length), :]
            self.validation = X, Y
        else :
            X, Y = self.validation
        return X, Y

    def get_test(self):
        if self.test == None:
            X = self.get_data(self.config.test_path + "-features.pkl")
            Y = self.get_data(self.config.test_path + "-labels.pkl")
            self.test = X, Y
        else:
            X, Y = self.test

        return X, Y

    def next_batch(self, data):
        if data.lower() not in ["train", "test", "validation"]:
            raise ValueError
        func = {"train" : self.get_train, "test": self.get_test, "validation": self.get_validation}[data.lower()]
        X, Y = func()
        start = 0
        batch_size = self.config.batch_size
        tot = len(X)
        total = int(tot/ batch_size) # fix the last batch
        while start < total:
            end = start + batch_size
            x = X[start : end, :]
            y = Y[start : end, :]
            start += 1
            yield (x, y, int(total))

    def eliminate_data(self, X, Y):
        """Eliminate data with one instaces in y is full 1 or full 0
        ----------
        X : numpy.ndarray
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndaarray {0,1}
            binary indicator matrix with label assignments.

        Returns
        -------
        numpy.ndarray size (n_new_samples, n_features)
        numpy.ndarray size (n_new_samples, n_labels)
        """
        num_labels = Y.shape[1]
        full_true = np.ones(num_labels)
        full_false = np.zeros(num_labels)

        i = 0
        while (i < len(Y)):
            if (Y[i] == full_true).all() or (Y[i] == full_false).all():
                Y = np.delete(Y, i, axis=0)
                X = np.delete(X, i, axis=0)
            else:
                i = i + 1
        return X, Y