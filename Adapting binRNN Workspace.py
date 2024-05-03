#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For the EN.625.692.81.SP24 Probabilistic Models Semester Project
By Shelby Golden April 27th, 2024

This is a workspace used to update the functions used in the CB153 sample
CRF analysis provided in the biRNN-CRF GitHub page. It is also used to better
understand the workflow associated with their analysis, so that it can be
adapted for the analysis done in this semester project.

One of the principal differences is this file uses TensorFlow2. Functions in 
this and the related binRNN_CRF_master folder are updated to reflect the 
differences between TensorFlow version 1 and version 2.


Source: https://github.com/alrojo/biRNN-CRF/blob/master/cb513.ipynb
Enviroment: conda activate "/Users/shelbygolden/miniconda3/envs/spyder-env" 
Spyder: /Users/shelbygolden/miniconda3/envs/spyder-env/bin/python
"""        
   

##########################################################
##########################################################
## package import and function definitions
#%matplotlib inline 
#%matplotlib nbagg     
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph

import biRNN_CRF_master.data as data
import biRNN_CRF_master.utils as utils
import biRNN_CRF_master.custom_ops as custom_ops
import biRNN_CRF_master.conditional_random_fields as crf


# the following import does not have an easy replacement
# from tensorflow.contrib.layers import fully_connected, batch_norm
# therefore, the following adaptions were done
import tf_slim
from tf_slim.layers import layers as _layers




# functions for the rest of the script

#initialize the variable
init_op = tf.compat.v1.global_variables_initializer()


def tensor_result(obj, shape_array):
    """
    Extrac value of a tensor object with the graph and array
    structure defined. Ex: np.random.rand(1, 1) for shape[1, 1]
    """
    
    with tf.compat.v1.Session() as sess:
        sess.run(init_op) #execute init_op
        #print the random values that we sample
        
        #rand_array = np.random.rand()
        print (sess.run(obj, feed_dict={obj: shape_array} ))

    #Close session
    #sess.close()



def chop_sequences(X, t, mask, length):
    max_len = int(np.floor(np.max(length) ))
    return X[:, :max_len], t[:, :max_len], mask[:, :max_len]



##########################################################
##########################################################
## optimized sequence partitioning
X_train = None


reset_default_graph()

# Defining model
num_iterations = 3e4
batch_size=64
number_inputs=42
number_outputs=8
seq_len=None#100 # max 700
learning_rate = 0.001


tf.compat.v1.disable_eager_execution()

# replace all tf.placeholder() with tf.compat.v1.placeholder()
# replace None in shape with 1


X_input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, number_inputs], name='X_input')
X_length = tf.compat.v1.placeholder(tf.int32, shape=[None,], name='X_length')
t_input = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='t_input')
t_input_hot = tf.one_hot(t_input, number_outputs)
t_mask = tf.compat.v1.placeholder(tf.float32, shape=[None, None], name='t_mask')

num_units_encoder = 400
num_units_l1 = 200
num_units_l2 = 200

l1 = _layers.fully_connected(X_input, num_units_l1)
l1 = tf.concat([X_input, l1], 2)


# change tf.nn.rnn_cell.GRUCell() to tf.keras.layers.GRUCell()


cell_fw = tf.keras.layers.GRUCell(num_units_encoder)
cell_bw = tf.keras.layers.GRUCell(num_units_encoder)
#enc_cell = tf.nn.rnn_cell.OutputProjectionWrapper(enc_cell, number_outputs)


# change tf.nn.bidirectional_dynamic_rnn() to tf.compat.v1.nn.bidirectional_dynamic_rnn()


enc_outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=l1,
                                                 sequence_length=X_length, dtype=tf.float32)
enc_outputs = tf.concat(enc_outputs, 2)

outputs = tf.reshape(enc_outputs, [-1, num_units_encoder*2])
l2 = _layers.fully_connected(outputs, num_units_l2)
l_f = _layers.fully_connected(l2, number_outputs, activation_fn=None)
l_g = _layers.fully_connected(l2, number_outputs**2, activation_fn=None)

batch_size_shp = tf.shape(enc_outputs)[0]
seq_len_shp = tf.shape(enc_outputs)[1]
l_f_reshape = tf.reshape(l_f, [batch_size_shp, seq_len_shp, number_outputs])

l_g_reshape = tf.reshape(l_g, [batch_size_shp, seq_len_shp, number_outputs**2])
l_g_reshape = tf.reshape(l_g_reshape, [batch_size_shp, seq_len_shp, number_outputs, number_outputs])

f = l_f_reshape
g = tf.slice(l_g_reshape, [0, 0, 0, 0], [-1, seq_len_shp-1, -1, -1])


# doesnt work
#g_prev = tf.slice(f, [0, 0, 0], [-1, seq_len_shp-1, -1])
#g_nxt = tf.slice(f, [0, 1, 0], [-1, -1, -1])
#g = tf.concat(2, [g_prev, g_nxt])
#g = tf.reshape(g, [tf.shape(g)[0], tf.shape(g)[1], tf.shape(f)[2], tf.shape(f)[2]])

# zeros
#g = tf.zeros(tf.shape(g))


nu_alp = crf.forward_pass(f, g, X_length)
nu_bet = crf.backward_pass(f, g, X_length)


prediction = crf.log_marginal(nu_alp, nu_bet)


rand_array = np.random.rand(1, 1, 1)
tensor_result(f, rand_array)







def loss_and_acc():
    # sequence_loss_tensor is a modification of TensorFlow's own sequence_to_sequence_loss
    # TensorFlow's seq2seq loss works with a 2D list instead of a 3D tensors
    loss = -crf.log_likelihood(t_input_hot, f, g, nu_alp, nu_bet, X_length)#custom_ops.sequence_loss(preds, t_input, t_mask)
    
    # if you want regularization
    #reg_scale = 0.00001
    #alternative? regularize = tf.contrib.layers.l2_regularizer(reg_scale)
    #regularize = tf.compat.v1.estimator.layers.l2_regularizer(reg_scale)
    #params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    #reg_term = sum([regularize(param) for param in params])
    #loss += reg_term
    
    # calculate accuracy
    # replaced tf.to_int32(tf.argmax(prediction, 2)) with ...
    # ... tf.cast(tf.argmax(prediction, 2), tf.int32)
    argmax = tf.cast(tf.argmax(prediction, 2), tf.int32)
    
    # replaced tf.to_float(tf.equal(argmax, t_input)) with ...
    # ... tf.cast(tf.argmax(prediction, 2), tf.int32)
    correct = tf.cast(tf.equal(argmax, t_input), tf.float32) * t_mask
    
    accuracy = tf.reduce_sum(correct) / tf.reduce_sum(t_mask)
    
    return loss, accuracy, argmax


loss, accuracy, predictions = loss_and_acc()


# use lobal step to keep track of our iterations
global_step = tf.Variable(0, name='global_step', trainable=False)

# pick optimizer, try momentum or adadelta
# replace tf.train.AdamOptimizer() with tf.optimizers.Adam()
optimizer = tf.optimizers.Adam(learning_rate)

# extract gradients for each variable


x = tf.constant(5.0)
with tf.GradientTape() as g:
  g.watch(x)
  with tf.GradientTape() as gg:
    gg.watch(x)
    y = x * x
  dy_dx = gg.gradient(y, x)  # dy_dx = 2 * x
d2y_dx2 = g.gradient(dy_dx, x)  # d2y_dx2 = 2
print(dy_dx)



with tf.GradientTape() as g:
    dy_dx = g.compute_gradients(loss, [f, g])
    

tf.GradientTape.gradient(loss, [f, g], )
grads_and_vars = optimizer.compute_gradients(loss, var_list=[f, g], tape=tf.GradientTape(persistent=True))

# add below for clipping by norm
#gradients, variables = zip(*grads_and_vars)  # unzip list of tuples
#clipped_gradients, global_norm = (
#    tf.clip_by_global_norm(gradients, self.clip_norm) )
#grads_and_vars = zip(clipped_gradients, variables)
# apply gradients and make trainable function
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)






if X_train is None:
    X_train, X_valid, t_train, t_valid, mask_train, mask_valid, length_train, length_valid, num_seq_train =\
    data.get_train(seq_len)
    X_valid, t_valid, mask_valid = chop_sequences(X_valid, t_valid, mask_valid, length_valid)
    print("X_train,", X_train.shape, X_train.dtype)
    print("t_train,", t_train.shape, t_train.dtype)
    print("mask_train,", mask_train.shape, mask_train.dtype)
    print("length_train,", length_train.shape, length_train.dtype)
    print("num_seq_train", num_seq_train)
    print("X_valid,", X_valid.shape, X_valid.dtype)
    print("t_valid,", t_valid.shape, t_valid.dtype)
    print("mask_valid,", mask_valid.shape, mask_valid.dtype)
    print("length_valid,", length_valid.shape, length_valid.dtype)




# restricting memory usage, TensorFlow is greedy and will use all memory otherwise
gpu_opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
# initialize the Session
sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options=gpu_opts))


# test train part
sess.run(tf.compat.v1.global_variables_initializer())
feed_dict = {X_input: X_valid, X_length: length_valid, t_input: t_valid,
             t_mask: mask_valid}
fetches = [f, g, loss]


res = sess.run(fetches=fetches, feed_dict=feed_dict)


tensor_result(fetches, feed_dict)


print("f", res[0].shape)
print "g", res[1].shape
#print "y_i", res[2][0].shape
#print "y_plus", res[2][1].shape
print "log_likelihood", res[2].shape


