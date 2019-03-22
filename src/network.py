import sys
import utils
import numpy as np
import tensorflow as tf
from config import Config

# Source : https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)  
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

class Network(object):
    def __init__(self, config, summarizer):
        tf.set_random_seed(0)
        self.summarizer = summarizer
        self.config = config
        self.Wx1, self.Wx2, self.Wx3, self.bx1, self.bx2, self.bx3 = self.init_Fx_variables()
        self.We1, self.We2, self.be1, self.be2 = self.init_Fe_variables()
        self.Wd1, self.Wd2, self.bd1, self.bd2 = self.init_Fd_variables()

    def weight_variable_leaky_relu(self, shape, name):
        """
        leaky_relu weight init
        :param shape:
        :param name:
        :return:
        """
        w = tf.get_variable(name=name, shape=shape, initializer=tf.random_uniform_initializer)
        w = 2 * (w - 0.5) * 0.01
        return w

    def bias_variable_leaky_relu(self, shape, name):
        """
        leaky_relu bias init
        :param shape:
        :param name:
        :return:
        """
        b = tf.get_variable(name=name, shape=shape, initializer=tf.random_uniform_initializer)
        b = b * 0.1
        return b

    def weight_variable_sigmoid(self, shape, name):
        """
        sigmoid weight init
        :param shape:
        :param name:
        :return:
        """
        w = tf.get_variable(name=name, shape=shape, initializer=tf.random_uniform_initializer)
        w = 8 * (w - 0.5) * np.sqrt(6) / np.sqrt(shape[0]+shape[1])
        return w

    def bias_variable_sigmoid(self, shape, name):
        """
        sigmoid bias init
        :param shape:
        :param name:
        :return:
        """
        b = tf.get_variable(name=name, shape=shape, initializer=tf.random_uniform_initializer)
        b = b * 0
        return b

    def init_Fx_variables(self):
        """
        init Fx weight and bias
        :return:
        """
        W1 = self.weight_variable_leaky_relu([self.config.features_dim, self.config.solver.hidden_dim], "weight_x1")
        W2 = self.weight_variable_leaky_relu([self.config.solver.hidden_dim, self.config.solver.hidden_dim], "weight_x2")
        W3 = self.weight_variable_sigmoid([self.config.solver.hidden_dim, self.config.solver.latent_embedding_dim], "weight_x3")
        b1 = self.bias_variable_leaky_relu([self.config.solver.hidden_dim], "bias_x1")
        b2 = self.bias_variable_leaky_relu([self.config.solver.hidden_dim], "bias_x2")
        b3 = self.bias_variable_sigmoid([self.config.solver.latent_embedding_dim], "bias_x3")
        return W1, W2, W3, b1, b2, b3

    def init_Fe_variables(self):
        """
        init Fe weight and bias
        :return:
        """
        W1 = self.weight_variable_leaky_relu([self.config.labels_dim, self.config.solver.hidden_dim], "weight_e1")
        W2 = self.weight_variable_sigmoid([self.config.solver.hidden_dim, self.config.solver.latent_embedding_dim], "weight_e2")
        b1 = self.bias_variable_leaky_relu([self.config.solver.hidden_dim], "bias_e1")
        b2 = self.bias_variable_sigmoid([self.config.solver.latent_embedding_dim], "bias_e2")
        return W1, W2, b1, b2

    def init_Fd_variables(self):
        """
        init Fd weight and bias
        :return:
        """
        W1 = self.weight_variable_leaky_relu([self.config.solver.latent_embedding_dim, self.config.solver.hidden_dim], "weight_d1")
        W2 = self.weight_variable_sigmoid([self.config.solver.hidden_dim, self.config.labels_dim], "weight_d2")
        b1 = self.bias_variable_leaky_relu([self.config.solver.hidden_dim], "bias_d1")
        b2 = self.bias_variable_sigmoid([self.config.labels_dim], "bias_d2")
        return W1, W2, b1, b2

    def accuracy(self, y_pred, y):
        return tf.reduce_mean(tf.cast(tf.equal(tf.round(y_pred), y), tf.float32))

    def Fx(self, X, keep_prob):
        """
        caculate Fx
        :param X: placeholder {n_intances, n_features}
            feature data
        :param keep_prob: tensorflow placeholder
            for dropout
        :return: tensor {n_intances, n_latent_embedding_dim}
            Fx latent data
        """
        hidden1 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(X, self.Wx1) + self.bx1), keep_prob)
        hidden2 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(hidden1, self.Wx2) + self.bx2), keep_prob)
        hidden3 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(hidden2, self.Wx3) + self.bx3), keep_prob)
        hidden3 = tf.subtract(hidden3, 0.5)
        return hidden3

    def Fe(self, Y, keep_prob):
        """
        caculate Fe
        :param Y: placeholder {n_intances, n_labels}
            label data
        :param keep_prob: tensorflow placeholder
            for dropout
        :return: tensor {n_intances, n_latent_embedding_dim}
            Fe latent data
        """
        hidden1 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(Y, self.We1)) + self.be1, keep_prob)
        pred = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(hidden1, self.We2) + self.be2), keep_prob)
        pred = tf.subtract(pred, 0.5)
        return pred

    def Fd(self, input, keep_prob):
        """
        caculate Fd
        :param X: placeholder {n_intances, n_latent_embedding_dim}
            feature data
        :param keep_prob: tensorflow placeholder
            for dropout
        :return: tensor {n_intances, n_labels}
            Fd predict logits
        """
        hidden1 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(input, self.Wd1) + self.bd1), keep_prob)
        logits = tf.matmul(hidden1, self.Wd2) + self.bd2
        pred = tf.nn.sigmoid(logits)
        return pred

    def prediction(self, X, keep_prob):
        """
        for pairwise loss use Fd(Fx(x))
        :param Y: placeholder {n_intances, n_features}
        :param keep_prob: tensorflow placeholder
            for dropout
        :return: tensor {n_intances, n_labels}
            Fd predict logits
        """
        Fx = self.Fx(X, keep_prob)
        return self.Fd(Fx, keep_prob)

    def loss_prediction(self, Y, keep_prob):
        """
        for pairwise loss use Fd(Fe(x))
        :param Y: placeholder {n_intances, n_labels}
        :param keep_prob: tensorflow placeholder
            for dropout
        :return: tensor {n_intances, n_labels}
            Fd loss predict logits
        """
        Fe = self.Fe(Y, keep_prob)
        return self.Fd(Fe, keep_prob)

    def embedding_loss(self, Fx, Fe):
        """
        caculate embedding loss
        min(||Fx(X) - Fe(Y)||^2), subject to Fx(X)Fx(X)^T = Fe(Y)Fe(Y)^T = I
        use Lagrange method and lagrange coefficient equeal to 0.5
        :param Fx: tensor {n_intances, n_latent_embedding_dim}
            Fx latent embedding data
        :param Fe: tensor {n_intances, n_latent_embedding_dim}
            Fe latent embedding data
        :return: tensor
            all n_insances loss
        """
        I = tf.eye(tf.shape(Fx)[1])
        C1, C2, C3 = Fx - Fe, tf.matmul(tf.transpose(Fx), Fx) - I, tf.matmul(tf.transpose(Fe), Fe) - I
        loss = tf.trace(tf.matmul(C1, tf.transpose(C1))) + self.config.solver.lagrange_const * tf.trace(tf.matmul(C2, tf.transpose(C2)) + tf.matmul(C3, tf.transpose(C3)))
        return loss

    def output_loss(self, predictions, labels):
        """Computational error function,k属于Y，l属于Y补，计算ck - cl值，此时误差是对称的
        Parameters
        ----------
        y : tensorflow tensor {0,1}
            binary indicator matrix with label assignments.
        output : tensorflow tensor [0,1]
            neural network output value

        Returns
        -------
        tensorflow tensor
        """
        shape = tf.shape(labels)
        y_i = tf.equal(labels, tf.ones(shape))
        y_not_i = tf.equal(labels, tf.zeros(shape))

        # get indices to check
        truth_matrix = tf.to_float(self.pairwise_and(y_i, y_not_i))

        # calculate all exp'd differences
        # through and with truth_matrix, we can get all c_i - c_k(appear in the paper)
        sub_matrix = self.pairwise_sub(predictions, predictions)
        exp_matrix = tf.exp(tf.negative(5 * sub_matrix))

        # check which differences to consider and sum them
        sparse_matrix = tf.multiply(exp_matrix, truth_matrix)
        sums = tf.reduce_sum(sparse_matrix, axis=[1, 2])

        # get normalizing terms and apply them
        y_i_sizes = tf.reduce_sum(tf.to_float(y_i), axis=1)
        y_i_bar_sizes = tf.reduce_sum(tf.to_float(y_not_i), axis=1)
        normalizers = tf.multiply(y_i_sizes, y_i_bar_sizes)

        loss = tf.divide(sums, normalizers)
        zero = tf.zeros_like(loss)
        loss = tf.where(tf.logical_or(tf.is_inf(loss), tf.is_nan(loss)), x=zero, y=loss)
        loss = tf.reduce_sum(loss)
        return loss

    def pairwise_and(self, a, b):
        """compute pairwise logical and between elements of the tensors a and b
        Description
        -----
        if y shape is [3,3], y_i would be translate to [3,3,1], y_not_i is would be [3,1,3]
        and return [3,3,3],through the matrix ,we can easy to caculate c_k - c_i(appear in the paper)
        """
        column = tf.expand_dims(a, 2)
        row = tf.expand_dims(b, 1)
        return tf.logical_and(column, row)

    def pairwise_sub(self, a, b):
        """compute pairwise differences between elements of the tensors a and b
        :param a:
        :param b:
        :return:
        """
        column = tf.expand_dims(a, 2)
        row = tf.expand_dims(b, 1)
        return tf.subtract(column, row)

    def cross_loss(self, features, labels, keep_prob):
        predictions = self.prediction(features, keep_prob)
        Fx = self.Fx(features, keep_prob)
        Fe = self.Fe(labels, keep_prob)
        cross_loss = tf.add(tf.log(1e-10 + tf.nn.sigmoid(predictions)) * labels, tf.log(1e-10 + (1 - tf.nn.sigmoid(predictions))) * (1 - labels))
        cross_entropy_label = -1 * tf.reduce_mean(tf.reduce_sum(cross_loss, 1))
        return cross_entropy_label

    def loss(self, features, labels, keep_prob):
        """
        caculate all instances loss, inlcude output loss ,embedding loss and regularization loss
        :param features: placeholder {n_instances, n_features}
        :param labels: placeholder {n_instances, n_labels}
        :param keep_prob: placeholder
        :return: tensor
        """
        loss_prediction = self.loss_prediction(labels, keep_prob)
        Fx = self.Fx(features, keep_prob)
        Fe = self.Fe(labels, keep_prob)
        l2_norm = tf.reduce_sum(tf.square(self.Wx1)) + tf.reduce_sum(tf.square(self.Wx2)) + tf.reduce_sum(tf.square(self.Wx3)) + tf.reduce_sum(tf.square(self.We1)) + tf.reduce_sum(tf.square(self.We2)) + tf.reduce_sum(tf.square(self.Wd1)) + tf.reduce_sum(tf.square(self.Wd2))
        l2_norm = 1e-3 * l2_norm

        return self.embedding_loss(Fx, Fe) + self.config.solver.alpha * self.output_loss(loss_prediction, labels) + l2_norm

    def train_step(self, loss, lr):
        """
        choose optimizer for minimize loss, this use RMSProp
        :param loss: tensor
        :return: tensor
            optimizer
        """
        optimizer = self.config.solver.optimizer
        return optimizer(learning_rate=lr, decay=0.9, momentum=0.99).minimize(loss)
