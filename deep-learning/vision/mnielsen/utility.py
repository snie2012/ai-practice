import numpy as np

def sigmoid(z):
	# The sigmoid function
	return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
	# Derivative of the sigmoid function
	return sigmoid(z) * (1 - sigmoid(z))

def vectorized_result(j):
	"""
	Return a 10-dimensional unit vector with a 1.0 in the j'th position
	and zeroes elsewhere.  This is used to convert a digit (0...9)
	into a corresponding desired output from the neural network.
	"""
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e