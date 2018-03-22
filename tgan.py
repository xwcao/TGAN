#https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# A supporting script to make tensor layer
from tnsr import *

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from util import normal_init, get_number_parameters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Size of the batch
mb_size = 128

# Dimension of the prior
z_dim = 40

# Size of the hidden layer
h_dim = 40

# Learning Rate
lr = 5e-3

# The input X
X = tf.placeholder(tf.float32, shape=[None, 28, 28])

# Weight matrices and bias tensor for the first and second tensor layer for the discriminator D
D_U_00 = tf.Variable(normal_init([h_dim, 28]))
D_U_01 = tf.Variable(normal_init([h_dim, 28]))
D_b_0 = tf.Variable(tf.zeros(shape=[h_dim,h_dim]))

D_U_10 = tf.Variable(normal_init([1, h_dim]))
D_U_11 = tf.Variable(normal_init([1, h_dim]))
D_b_1 = tf.Variable(tf.zeros(shape=[1,1]))

# Parameters for discriminator
theta_D = [D_U_00, D_U_01, D_U_10, D_U_11, D_b_0, D_b_1]


# Prior Z
Z = tf.placeholder(tf.float32, shape=[None, z_dim, z_dim])

# Weight matrices and bias tensor forthe first and second tensor layer for the generator G
G_U_00 = tf.Variable(normal_init([h_dim, z_dim]))
G_U_01 = tf.Variable(normal_init([h_dim, z_dim]))
G_b_0 = tf.Variable(tf.zeros(shape=[h_dim,h_dim]))

G_U_10 = tf.Variable(normal_init([28, h_dim]))
G_U_11 = tf.Variable(normal_init([28, h_dim]))
G_b_1 = tf.Variable(tf.zeros(shape=[28,28]))

# Parameters for generator
theta_G = [G_U_00, G_U_01, G_U_10, G_U_11, G_b_0, G_b_1]




def sample_z(shape):
    return np.random.uniform(-1., 1., size=shape)


def generator(z):

    # First Tensorized layer
    out = tensor_layer(Z, [G_U_00, G_U_01], G_b_0, tf.nn.relu)

    # Second Tensorized layer
    out = tensor_layer(out, [G_U_10, G_U_11], G_b_1, tf.nn.sigmoid)

    return out


def discriminator(x):
    # First Tensorized layer
    out = tensor_layer(x, [D_U_00, D_U_01], D_b_0, tf.nn.relu)

    # Return the logit and prob reoresentation after sigmoid
    return tensor_layer(out, [D_U_10, D_U_11], D_b_1, tf.nn.sigmoid), tensor_layer(out, [D_U_10, D_U_11], D_b_1, identity)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

print("Total number of parameters: {}".format(get_number_parameters(theta_G+theta_D)))

G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(G_loss, var_list=theta_G)

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_z([16, z_dim, z_dim])})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb.reshape(mb_size, 28, 28), Z: sample_z([mb_size, z_dim, z_dim])})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_z([mb_size, z_dim, z_dim])})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print('')