# TGAN
Code for the paper "Tensorizing Generative Adversarial Nets"

# Tensor Layer
Given an input tensor X, we apply multilinear transformation to it, then we perform element-wise activations to form a tensor layer. For details, refer to the paper [[1]](https://arxiv.org/abs/1710.10772). The function 'tensor_layer(tensor, matrices, bias, activation_function)' in 'tnsr.py' defines the presented tensor layer which forms TGAN. Refer hyperparameter.png for hyperparameters used for the paper.

```python
def tensor_layer(tensor, matrices, bias, activation_function):
	"""
	INPUT: tensor, matrices, bias and activation function
		   tensor: tensorflow obejct 
		   matrices: list of matrix (tf object again)
		   bias: you know
		   activation function: function such as tf.nn.relu
	OUTPUT: tensor-layer
			mode-dot operation is applied and it changes
			the dimensions of the tensor by contraction operations
	"""
```

# Example
To run an example of tensorized GAN with MNIST dataset, run the following
```bash
python tgan.py
```

# Libraries
* Python 2.7.14
* Tensorflow 1.1.0
* Numpy 1.12.1
* Matplotlib 2.1.0

# Reference
[[1]](https://arxiv.org/abs/1710.10772) Cao, Xingwei, and Qibin Zhao. "Tensorizing Generative Adversarial Nets." arXiv preprint arXiv:1710.10772 (2017).