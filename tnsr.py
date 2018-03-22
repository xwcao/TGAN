import tensorflow as tf

def mode_dot(tensor, matrix, mode):
	"""
    INPUT:  tensor, matrix
            mode of the tensor to perform contranction (starts from 1)
    OUTPUT: tensor after contraction

    EXAMPLE:
   			Given a tensor X in a space 3x4x5 and a matrix of shape 6x4
   			We output a tensor X' of shape 3x6x5

			First we check the conditions to perform this operation
			Then, given mode (in the case of example, 2 beacuse 4 is second dimension of X), we make a list [[mode-1],[1]] and feed it to tensordot operation

			Because of how tensordot was written, we need to permutate the index of the tensor.
			In the case above, the shape of the output after tensordot is 3x5x6, which is not what we want so we permutate it to 3x6x5
			We make a list p (permutation) to transpose the index of the tensor X.

	We assume that the first mode of the tensor is for each sample (batch size)
    """
	assert len(matrix.get_shape().as_list()) == 2, "The order of matrix is not 2."

	order = len(tensor.get_shape().as_list())

	# hello python 3
	p = list(range(order-1))
	p.insert(mode, order-1)

	return tf.transpose(tf.tensordot(tensor,matrix,[[mode],[1]]), perm = p)


def tensor_layer(tensor, matrices, bias, activation_function):
	"""
	INPUT: tensor, matrices, bias and activation function
		   tensor: tensorflow obejct 
		   matrices: list of matrix (tf object again)
		   bias: you know
		   activation function: function such as tf.nn.relu
	OUTPUT: tensor-layer
			mode-dot operation is applied and it changes the dimensions of the tensor by contraction operations

	We assume that the first mode of the tensor is for each sample (batch size)
	"""

	assert len(matrices) == len(tensor.get_shape().as_list())-1, "The length of list of matrices has to match the length of the input tensor."

	for i in range(len(matrices)):
		tensor = mode_dot(tensor, matrices[i], i+1)

	return activation_function(tensor) if bias == None else activation_function(tensor+bias)


def leakyReLU(tensor):
	return tf.maximum(tensor, 0.2 * tensor)

def identity(tensor):
	return tensor