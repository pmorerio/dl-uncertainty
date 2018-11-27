import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os	

#~ from utils import resize_images

class Solver(object):

    def __init__(self, model, batch_size=128, train_iter=250000, 
		    mnist_dir='../mnist', log_dir='logs',
		    model_save_path='model', trained_model='model/model'):
        
        self.model = model
	#actually builds the graph
	self.model.build_model()
	
        self.batch_size = batch_size
        self.train_iter = train_iter
	self.mnist_dir = mnist_dir
        self.log_dir = log_dir
        self.model_save_path = model_save_path
        self.trained_model = model_save_path + '/model'
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
	self.config.allow_soft_placement = True


    def load_mnist(self, image_dir, split='train'):
        print ('[*] Loading MNIST dataset.')
	
	image_file = 'train.pkl' if split=='train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            mnist = pickle.load(f)
        images = mnist['X'] / 127.5 - 1
        labels = mnist['y']
	
        return images, np.squeeze(labels).astype(int)
	
	
    def train(self):
	
	# make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
	
	print '[*] Training.'
	
	images, labels = self.load_mnist(self.mnist_dir, split='train')
	
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
	    
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
	   
	    print ('Start training.')
	    count = 0
	    t = 0

	    for step in range(self.train_iter):
		
		count += 1
		t+=1
		
		i = step % int(images.shape[0] / self.batch_size)
	
		       
		feed_dict = {self.model.images: images[i*self.batch_size:(i+1)*self.batch_size]}  
		
		sess.run(self.model.train_op, feed_dict) 

		if t%500==0 or t==1:

		    summary, l,  = sess.run([self.model.summary_op, self.model.loss], feed_dict)
		    summary_writer.add_summary(summary, t)
		    print ('Step: [%d/%d]  loss: [%.6f] ' \
			       %(t, self.train_iter, l))
		    saver.save(sess, os.path.join(self.model_save_path, 'model'))
	    

    def test(self, checkpoint):
	
	print '[*] Test.'
	
	images, labels = self.load_mnist(self.mnist_dir, split='test')
	
        with tf.Session(config=self.config) as sess:
	    
	    variables_to_restore = slim.get_model_variables(scope='encoder')
	    variables_to_restore += slim.get_model_variables(scope='decoder')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, checkpoint)
	    
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir+'/test', graph=tf.get_default_graph())

	    print ('[*] Start testing.')

	    num_batches = int(images.shape[0] / self.batch_size)
	    
	    for i in range(num_batches):
		       
		feed_dict = {self.model.images: images[i*self.batch_size:(i+1)*self.batch_size]}  

		summary, l,  = sess.run([self.model.summary_op, self.model.loss], feed_dict)
		summary_writer.add_summary(summary, i)
		print ('Batch: [%d/%d]  loss: [%.6f] ' \
			   %(i, num_batches, l))

		    
if __name__=='__main__':

    print('empty')
