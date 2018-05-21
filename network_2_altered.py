# -*- coding: utf-8 -*-


"""
network.py
~~~~~~~~~~
credit for original code goes to mnielsen (https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py)
code updated for python 3.6
"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5*np.linalg.norm(a-y)**2
    
    @staticmethod
    def batch_fn(a_matrix, y_matrix):
        """Return the mean cost associated with an output matrix ``a_matrix`` 
        and desired output matrix ``y_matrix``.
        """
        return 1/(2*a_matrix.shape[1])*(np.linalg.norm(a_matrix-y_matrix))**2
        

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    @staticmethod
    def batch_fn(a_matrix, y_matrix):
        """Return the mean cost associated with an output matrix ``a_matrix`` 
        and desired output matrix ``y_matrix``.
        """
        return 1/(a_matrix.shape[1])*np.sum(np.nan_to_num(-y_matrix*np.log(a_matrix)-(1-y_matrix)*np.log(1-a_matrix)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a-y)



#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def batch_ff(self, mini_batch):
        """Return an output matrix of the network if mini_batch is a list of 
        tuples (datapoint, ground_truth)
        """
        len_datapt = len(mini_batch[0][0])
        len_mini_batch = len(mini_batch)
        a_matrix = np.zeros((len_datapt, len_mini_batch))
        for index, item in enumerate(mini_batch):
            a_matrix[:,index] = item[0][:,0]
        for b, w in zip(self.biases, self.weights):
            b = np.tile(b, len_mini_batch)
            a_matrix = sigmoid(np.dot(w, a_matrix)+b)
        return a_matrix

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.
        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print("Epoch {} training complete".format(j))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))
            print()
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        nabla_b, nabla_w = self.batch_backprop(mini_batch)
        self.weights = [(1-eta*(lmbda/n))*w-(eta)*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta)*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def batch_backprop(self, mini_batch):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        """
        len_datapt = len(mini_batch[0][0])
        len_mini_batch = len(mini_batch)
        len_y = len(mini_batch[0][1])
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # initialize x_matrix and y_matrix
        x_matrix = np.zeros((len_datapt, len_mini_batch))
        y_matrix = np.zeros((len_y, len_mini_batch))
        for index, item in enumerate(mini_batch):
            x_matrix[:,index] = item[0][:,0]
            y_matrix[:,index] = item[1][:,0]
        # feedforward
        activation = x_matrix
        activations = [x_matrix] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            b = np.tile(b, len_mini_batch)
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta_matrix = (self.cost).delta(zs[-1], activations[-1], y_matrix)
        delta_avg = np.reshape(np. average(delta_matrix, axis=1), (delta_matrix.shape[0], 1))
        
        nabla_b[-1] = delta_avg
        nabla_w_matrices = []
        for i in range(len(mini_batch)):
            nabla_w_matrices.append(np.dot(np.reshape(delta_matrix[:,i], (delta_matrix.shape[0], 1)),\
                                           np.reshape(activations[-2][:,i], (activations[-2].shape[0],1)).transpose()))
        nabla_w[-1] = sum(nabla_w_matrices)/len_mini_batch
        # backpropagation
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta_matrix = np.dot(self.weights[-l+1].transpose(), delta_matrix) * sp
            delta_avg = np.reshape(np. average(delta_matrix, axis=1), (delta_matrix.shape[0], 1))
            nabla_b[-l] = delta_avg
            nabla_w_matrices = []
            for i in range(len(mini_batch)):
                nabla_w_matrices.append(np.dot(np.reshape(delta_matrix[:,i], (delta_matrix.shape[0], 1)),\
                                        np.reshape(activations[-l-1][:,i], (activations[-l-1].shape[0],1)).transpose()))
            nabla_w[-l] = sum(nabla_w_matrices)/len_mini_batch
        return (nabla_b, nabla_w)
        

    def accuracy(self, mini_batch, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.
        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.
        """
        if convert:
            len_mini_batch = len(mini_batch)
            len_y = len(mini_batch[0][1])
            y_matrix = np.zeros((len_y, len_mini_batch))
            for index, item in enumerate(mini_batch):
                y_matrix[:,index] = item[1][:,0]
            results = zip(np.argmax(self.batch_ff(mini_batch), axis=0), np.argmax(y_matrix, axis=0))
        else:
            y_list = [y for x,y in mini_batch]
            results = zip(np.argmax(self.batch_ff(mini_batch), axis=0), y_list)
        return sum(int(x == y) for (x, y) in results)

#    def total_cost(self, data, lmbda, convert=False):
#        """Return the total cost for the data set ``data``.  The flag
#        ``convert`` should be set to False if the data set is the
#        training data (the usual case), and to True if the data set is
#        the validation or test data.  See comments on the similar (but
#        reversed) convention for the ``accuracy`` method, above.
#        """
#        cost = 0.0
#        for x, y in data:
#            a = self.feedforward(x)
#            if convert: y = vectorized_result(y)
#            cost += self.cost.fn(a, y)/len(data)
#        cost += 0.5*(lmbda/len(data))*sum(
#            np.linalg.norm(w)**2 for w in self.weights)
#        return cost
    
    def total_cost(self, mini_batch, lmbda, convert=False):
        len_mini_batch = len(mini_batch)
        if convert:
            len_y = len(vectorized_result(mini_batch[0][1]))
        else:
            len_y = len(mini_batch[0][1])
        a_matrix = self.batch_ff(mini_batch)
        y_matrix = np.zeros((len_y, len_mini_batch))
        for index, item in enumerate(mini_batch):
            if convert:
                y_matrix[:,index] = vectorized_result(item[1])[:,0]
            else:
                y_matrix[:,index] = item[1][:,0]
        cost = self.cost.batch_fn(a_matrix, y_matrix)
        cost += 0.5*(lmbda/len_mini_batch)*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        
    def save_weights(self, filename):
        """Save the neural network weights to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        
    def load_weights(self, filename):
        """Save the neural network weights from the file ``filename``."""
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        self.weights = [np.array(w) for w in data["weights"]]
        self.biases = [np.array(b) for b in data["biases"]]
        
#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))










