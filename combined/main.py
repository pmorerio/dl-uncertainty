import tensorflow as tf
from model import Model
from solver import Solver

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'train', or 'test'")
flags.DEFINE_string('model_save_path', 'model0', "base directory for saving the models")
flags.DEFINE_string('device', '/gpu:0', "/gpu:id number")
flags.DEFINE_string('checkpoint', './model/model0/model', "Model checkpoint to be tested")
FLAGS = flags.FLAGS

def main(_):
    
    with tf.device(FLAGS.device):
	
	model_save_path = 'model/'+FLAGS.model_save_path	
	# create directory if it does not exist
	if not tf.gfile.Exists(model_save_path):
		tf.gfile.MakeDirs(model_save_path)
	log_dir = 'logs/'+ model_save_path
	
	model = Model(learning_rate=0.0003, mode=FLAGS.mode)
	solver = Solver(model, model_save_path=model_save_path, log_dir=log_dir)
	
	# create directory if it does not exist
	if not tf.gfile.Exists(model_save_path):
		tf.gfile.MakeDirs(model_save_path)
	
	if FLAGS.mode == 'train':
		solver.train()
	elif FLAGS.mode == 'test':
		solver.test(checkpoint=FLAGS.checkpoint)
	else:
	    print 'Unrecognized mode.'
        
if __name__ == '__main__':
    tf.app.run()



    


