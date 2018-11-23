import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np


class Model(object):

    def __init__(self, mode='train', hidden_size = 128, learning_rate=0.0001, batch_size=128):
        
	self.mode=mode
	self.learning_rate = learning_rate
	self.hidden_repr_size = hidden_size
	self.batch_size = batch_size
	self.test_trials = 20
    
    #  make sure dropout is ALWAYS on 
    def EncoderDecoder(self, images, is_training = True, reuse=False):
	
	if images.get_shape()[3] == 3:
	    images = tf.image.rgb_to_grayscale(images)
	
	with tf.variable_scope('encoder',reuse=reuse):
	    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
		with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):
		    net = slim.conv2d(images, 64, 5, scope='conv1')
		    net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
		    net = slim.conv2d(net, 128, 5, scope='conv2')
		    net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
		    net = tf.contrib.layers.flatten(net)
		    net = slim.dropout(net, 0.5, is_training=is_training)
		    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc3')
		    net = slim.dropout(net, 0.5, is_training=is_training)
		    net = slim.fully_connected(net, self.hidden_repr_size, activation_fn=tf.tanh,scope='fc4')
		    net = slim.dropout(net, 0.5, is_training=is_training)

	net = tf.expand_dims(net, 1)
	net = tf.expand_dims(net, 1)
	
	with tf.variable_scope('decoder', reuse=reuse):
	    with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None,           
				    stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
		with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
					activation_fn=tf.nn.relu, is_training=is_training):

		    net = slim.conv2d_transpose(net, 256, [7, 7], padding='VALID', scope='conv_transpose1')   # (batch_size, 7, 7, 128)
		    net = slim.batch_norm(net, scope='bn1_gen')
		    net = slim.dropout(net, 0.5, is_training=is_training)
		    mean = slim.conv2d_transpose(net, 256, [3, 3], scope='conv_transpose2_mean')   # (batch_size, 14, 14, 128)
		    mean = slim.batch_norm(mean, scope='bn2_gen_mean')
		    mean = slim.conv2d_transpose(mean, 1, [3, 3], activation_fn=tf.tanh, scope='conv_transpose3_mean')   # (batch_size, 28, 28, 1)
		    
	return mean
    
		    
    def build_model(self):
	
	print('[*] Building model')
	self.images = tf.placeholder(tf.float32, [None, 28, 28, 1], 'images')
	self.rec_images = self.EncoderDecoder(self.images)
	
	# sample N sub-nets and average
	if self.mode == 'test':
	    self.rec_images = tf.expand_dims(self.rec_images,0)
	    for i in range(self.test_trials):
		self.rec_images = tf.concat([self.rec_images, tf.expand_dims( self.EncoderDecoder(self.images, reuse=True), 0) ], axis=0 )
	    self.mean, self.var = tf.nn.moments(self.rec_images, axes=[0])
	    
	    # summary op
	    image_summary = tf.summary.image('images', self.images)
	    rec_image_summary = tf.summary.image('rec_images', self.mean)
	    var_summary = tf.summary.image('epistemic_uncertaintiy', self.var)
	    
	    self.summary_op = tf.summary.merge([image_summary, \
						rec_image_summary, \
						var_summary]
						)
	    
	

	if self.mode == 'train':
	    # loss
	    self.loss = tf.reduce_mean( tf.square( self.rec_images - self.images ) ) 
	    # training stuff
	    self.optimizer = tf.train.AdamOptimizer(self.learning_rate) 
	    self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
	
	    # summary op
	    loss_summary = tf.summary.scalar('loss', self.loss)
	    image_summary = tf.summary.image('images', self.images)
	    rec_image_summary = tf.summary.image('rec_images', self.rec_images)
	    
	    self.summary_op = tf.summary.merge([loss_summary, \
						image_summary, \
						rec_image_summary])
    
