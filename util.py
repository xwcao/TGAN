import numpy as np
import tensorflow as tf

weight_std = 0.2

def next_batch(num, data):
    '''
    Return a random batch of samples. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]

    return np.asarray(data_shuffle)

def normal_init(size):
    return tf.random_normal(shape = size, stddev = weight_std)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def get_number_parameters(vars):
	total = 0
	for var in vars:
		total += np.prod(var.get_shape().as_list())
	return total