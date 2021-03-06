"""
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py

An improved version of network1.py, implementing the stochastic gradient decent 
learning algorithm fora feed forward neural network. Improvements include the addition 
of the cross-entropy cost function, regularization, and better initialization of network
weights.
"""

import json
import random
import sys

import numpy as np
from utility import sigmoid, sigmoid_prime, vectorized_result

class QuadraticCost(object):

	@staticmethod
	def fn(a, y):
		"""
		return the cost associated with an output `a` and desired output `y`
		"""
		return 0.5 * np.linalg.norm(a-y)**2

	@staticmethod
	def delta(z, a, y):
		"""
		return the error delta from the output layer. 
		"""
		return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):

	@staticmethod
	def fn(a, y):
		"""
		Return the cost associated with an output `a` and desired output
		`y`. Note that np.nan_to_num is used to ensure numerical stability.
		In particular, if both `a` and `y` have a 1.0 in the same slot, 
		then the expression (1-y)*np.log(1-a) returns nan. The np.nan_to_num
		ensures that that is converted to the correct value (0.0).
		"""
		return np.sum(np.nan_to_num(-y * np.log(a) - (1-y) * np.log(1-a)))

	@staticmethod
	def delta(z, a, y):
		"""
		Return the error delta from the output layer. Note that the parameter `z`
		is not used by the method. It is included in the method's paramenters in 
		order to make the interface cosistent with the delta method for other cost 
		classes
		"""
		return (a - y)


class Network(object):

	def __init__(self, sizes, cost=CrossEntropyCost):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.default_weight_initializer()
		self.cost = cost

	def default_weight_initializer(self):
		"""
		Initialize each weight using a Gaussian distribution with mean 0
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
		self.weights = [np.random.randn(y, x) / np.sqrt(x)
						for x, y in zip(self.sizes[:-1], self.sizes[1:])] 

	def large_weight_initializer(self):
		"""
		Initialize each weight using a Guassian distribution with mean 0
		and standard diviation 1. Initialize the biases using a Guassian
		distribution with mean 0 and standard deviation 1.

		Note that the first layer is assumed to be an input layer, and by
		convention we won't set any biases for those neurons, since biases
		are only ever used in computing the outputs from later layers.

		This weights and bias initializer uses the same approach as in 
		network1.py, and is included for purposes of comparison. It will 
		usually be better to use the default weight initializer instead.
		"""
		self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y, x)
						for x, y in zip(self.sizes[:-1], self.sizes[1:])]

	def feedforward(self, a):
		# Return the output of the network is `a` is the input
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta,
			lmbda=0.0,
			evaluation_data=None,
			monitor_evaluation_cost=False,
			monitor_evaluation_accuracy=False,
			monitor_training_cost=False,
			monitor_training_accuracy=False):
		if evaluation_data: n_data = len(evaluation_data)
		n = len(training_data)
		evaluation_cost, evaluation_accuracy = [], []
		training_cost, training_accuracy = [], []
		for j in xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [
					training_data[k:k+mini_batch_size]
					for k in xrange(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(
					mini_batch, eta, lmbda, len(training_data))
			print "Epoch %s training complete" % j
			if monitor_training_cost:
				cost = self.total_cost(training_data, lmbda)
				training_cost.append(cost)
				print "Cost on training data: {}".format(cost)
			if monitor_training_accuracy:
				accuracy = self.accuracy(training_data, convert=True)
				training_accuracy.append(accuracy)
				print "Accuracy on training data: {} / {}".format(
					accuracy, n)
			if monitor_evaluation_cost:
				cost = self.total_cost(evaluation_data, lmbda, convert=True)
				evaluation_cost.append(cost)
				print "Cost on evaluation data: {}".format(cost)
			if monitor_evaluation_accuracy:
				accuracy = self.accuracy(evaluation_data)
				evaluation_accuracy.append(accuracy)
				print "Accuracy on evaluation data: {} / {}".format(
					self.accuracy(evaluation_data), n_data)
				print
		   
		return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

	def update_mini_batch(self, mini_batch, eta, lmbda, n):
		nabla_b = [np.zeros_like(b) for b in self.biases]
		nabla_w = [np.zeros_like(w) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [(1 - eta*(lmbda/n)) * w - (eta/len(mini_batch)) * nw
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b - (eta / len(mini_batch)) * nb
						for b, nb in zip(self.biases, nabla_b)]

	def backprop(self, x, y):
		nabla_b = [np.zeros_like(b) for b in self.biases]
		nabla_w = [np.zeros_like(w) for w in self.weights]
		
		# feedforward
		activation = x
		activations = [x]
		zs = []
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		
		# backprop pass
		delta = self.cost.delta(zs[-1], activations[-1], y)
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].T)
		for l in xrange(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].T, delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].T)

		return (nabla_b, nabla_w)

	def accuracy(self, data, convert=False):
		"""
		Return the number of inputs in ``data`` for which the neural
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
			results = [(np.argmax(self.feedforward(x)), np.argmax(y))
						for (x, y) in data]
		else:
			results = [(np.argmax(self.feedforward(x)), y)
						for (x, y) in data]
		return sum(int(x==y) for (x, y) in results)

	def total_cost(self, data, lmbda, convert=False):
		"""
		Return the total cost for the data set ``data``.  The flag
		``convert`` should be set to False if the data set is the
		training data (the usual case), and to True if the data set is
		the validation or test data.  See comments on the similar (but
		reversed) convention for the ``accuracy`` method, above.
		"""
		cost = 0.0
		for x, y in data:
			a = self.feedforward(x)
			if convert: y = vectorized_result(y)
			cost += self.cost.fn(a, y) / len(data)
		cost += 0.5*(lmbda/len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)
		return cost

	def save(self, filename):
		data = {
			"sizes": self.sizes,
			"weights": [w.tolist() for w in self.weights],
			"biases": [b.tolist() for b in self.biases],
			"cost": str(self.cost.__name__)
		}
		f = open(filename, 'w')
		json.dump(data, f)
		f.close()


# load a network
def load(filename):
	"""
	Load a neural network from the file ``filename``.  Returns an
	instance of Network.
	"""
	with open(filename, "r") as f:
		data = json.load(f)
	cost = getattr(sys.modules[__name__], data["cost"])
	net = Network(data["sizes"], cost=cost)
	net.weights = [np.array(w) for w in data["weights"]]
	net.biases = [np.array(b) for b in data["biases"]]
	return net