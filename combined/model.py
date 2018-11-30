import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np


class Model(object):

    def __init__(self, mode='train', hidden_size = 128, learning_rate=0.0003, batch_size=128):
        
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
	    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
		net = slim.conv2d(images, 64, 3, scope='conv1') 
		net = slim.max_pool2d(net, 2, stride=2, scope='pool1') # [None, 14,14,64]
		net = slim.conv2d(net, 128, 3, scope='conv2') 
		net = slim.max_pool2d(net, 2, stride=2, scope='pool2') # [None, 7,7,128]
		
		net = slim.conv2d(net, 1024, 7,padding='VALID', scope='conv3') #[None, 1,1,1024]
		net = slim.dropout(net, 0.5, is_training=is_training)
		net = slim.conv2d(net, 1024, 1,padding='VALID', scope='conv4') #[None, 1,1,1024]
		net = slim.dropout(net, 0.5, is_training=is_training)
	
	with tf.variable_scope('decoder', reuse=reuse):
	    with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None,           
				    stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
		with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
					activation_fn=tf.nn.relu, is_training=(self.mode=='train')):
		    with slim.arg_scope([slim.conv2d], activation_fn=None, padding='SAME'):
			net = slim.conv2d_transpose(net, 512, [7, 7], padding='VALID', scope='conv_transpose1')   # (batch_size, 7, 7, 128)
			#~ net = slim.batch_norm(net, scope='bn1_gen')
			net = slim.dropout(net, 0.5, is_training=is_training)
			
			mean = slim.conv2d_transpose(net, 256, [3, 3], scope='conv_transpose1_mean')   # (batch_size, 14, 14, 256)
			mean = slim.batch_norm(mean, scope='bn1_mean')
			mean = slim.conv2d(mean, 128, [3, 3], activation_fn=None, scope='conv_1_mean')   # (batch_size, 14, 14, 128)
			mean = slim.batch_norm(mean, scope='bn2_mean')
			mean = slim.conv2d_transpose(mean, 64, [3, 3],  scope='conv_transpose_out_mean')   # (batch_size, 28, 28, 64)
			mean = slim.batch_norm(mean, scope='bn3_mean')
			mean = slim.conv2d(mean, 3, [3, 3], scope='conv_out_mean')   # (batch_size, 28, 28, 3)
			
			log_var = slim.conv2d_transpose(net, 256, [3, 3], scope='conv_transpose1_log_var')   # (batch_size, 14, 14, 256)
			log_var = slim.batch_norm(log_var, scope='bn1_log_var')
			log_var = slim.conv2d(log_var, 128, [3, 3], activation_fn=None, scope='conv_1_log_var')   # (batch_size, 14, 14, 128)
			log_var = slim.batch_norm(log_var, scope='bn2_log_var')
			log_var = slim.conv2d_transpose(log_var, 64, [3, 3],  scope='conv_transpose_out_log_var')   # (batch_size, 28, 28, 64)
			log_var = slim.batch_norm(log_var, scope='bn3_log_var')
			log_var = slim.conv2d(log_var, 3, [3, 3], scope='conv_out_log_var')   # (batch_size, 28, 28, 3)
		    
	return mean, log_var
    
		    
    def build_model(self):
	
	print('[*] Building model')
	self.images = tf.placeholder(tf.float32, [None, 28, 28, 1], 'images')
	self.rec_images, self.log_var = self.EncoderDecoder(self.images)
	
	# sample N sub-nets and average
	if self.mode == 'test':
	    self.rec_images = tf.expand_dims(self.rec_images,0)
	    for i in range(self.test_trials):
		self.rec_images = tf.concat([self.rec_images, tf.expand_dims( self.EncoderDecoder(self.images, reuse=True), 0)[0] ], axis=0 )
	    _ , self.var = tf.nn.moments(self.rec_images, axes=[0])
	    
	    # 1 forward pass with no dropout for aleatoric_uncertainty
	    self.rec_images2, self.log_var2 = self.EncoderDecoder(self.images, reuse=True, is_training=False)
	    
	    # summary op
	    image_summary = tf.summary.image('images', self.images)
	    rec_image_summary = tf.summary.image('rec_images', self.rec_images2)
	    uncertainty_summary = tf.summary.image('aleatoric_uncertainty', tf.exp(self.log_var2))
	    var_summary = tf.summary.image('epistemic_uncertaintiy', self.var)
	    
	    self.summary_op = tf.summary.merge([image_summary, \
						rec_image_summary, \
						var_summary,\
						uncertainty_summary])

	if self.mode == 'train':
	    # loss
	    self.loss1 = tf.reduce_mean( tf.exp(-self.log_var) * tf.square( (self.rec_images-self.images) ) )
	    self.loss2 = tf.reduce_mean(self.log_var)
	    # "From What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" NIPS 2017
	    # In practice, we train the network to predict the log variance
	    self.loss = .5*(self.loss1+self.loss2)

	
	    # training stuff
	    self.optimizer = tf.train.AdamOptimizer(self.learning_rate) 
	    self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
	
	    # summary op
	    loss_summary = tf.summary.scalar('loss', self.loss)
	    mean_summary = tf.summary.scalar('loss1', self.loss1)
	    log_var_summary = tf.summary.scalar('loss2', self.loss2)
	    image_summary = tf.summary.image('images', self.images)
	    rec_image_summary = tf.summary.image('rec_images', self.rec_images)
	    uncertainty_summary = tf.summary.image('aleatoric_uncertainty', tf.exp(self.log_var))
	    
	    self.summary_op = tf.summary.merge([loss_summary, \
						mean_summary, \
						log_var_summary, \
						image_summary, \
						rec_image_summary, \
						uncertainty_summary])
    
