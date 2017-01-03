import tensorflow as tf
import numpy as np
import os
from scipy import ndimage
from config import init_op, SummaryWriter


class Solver(object):
    """Load dataset and train DCGAN"""
    
    def __init__(self, model, num_epoch=10, image_path='data/celeb_resized', model_save_path='model/', log_path='log/'):
        self.model = model
        self.num_epoch = num_epoch
        self.image_path = image_path
        self.model_save_path = model_save_path
        self.log_path = log_path
        
        # create directory if not exists
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        # construct the dcgan model
        model.build_model()
        
        
    def load_dataset(self, image_path):
        print ('loading image dataset..')
        image_files = os.listdir(image_path)
        images = np.array(list(map(lambda x: ndimage.imread(os.path.join(image_path, x), mode='RGB'), image_files))).astype(np.float32)
        images = images / 127.5 - 1
        print ('finished loading image dataset..!')
        return images
    
    
    def train(self):
        model=self.model
        
        #load image dataset
        data = self.load_dataset(self.image_path)
        num_iter_per_epoch = int(data.shape[0] / model.batch_size)
        
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # initialize parameters 
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()
            
            # tensorboard
            summary_writer = SummaryWriter(logdir=self.log_path, graph=tf.get_default_graph())
             
            for e in range(self.num_epoch):
                for i in range(num_iter_per_epoch):
                    # train the discriminator
                    image_batch = data[i*model.batch_size:(i+1)*model.batch_size]
                    z_batch = np.random.uniform(-1, 1, size=[model.batch_size , model.dim_z])
                    feed_dict = {model.images: image_batch, model.z: z_batch}
                    sess.run(model.d_optimizer, feed_dict)
                    
                    # train the generator
                    feed_dict = {model.z: z_batch}
                    sess.run(model.g_optimizer, feed_dict)
                    
                    # train the generator twice to stabilize traininig (different from paper)
                    sess.run(model.g_optimizer, feed_dict)
                    
                    if i % 10 == 0:
                        feed_dict = {model.images: image_batch, model.z: z_batch}
                        summary, d_loss, g_loss = sess.run([model.summary_op, model.d_loss, model.g_loss], feed_dict)
                        summary_writer.add_summary(summary, e*num_iter_per_epoch + i)
                        print ('Epoch: [%d] Step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]' %(e+1, i+1, num_iter_per_epoch, d_loss, g_loss))
                        
                    if i % 500 == 0:  
                        model.saver.save(sess, os.path.join(self.model_save_path, 'dcgan-%d' %(e+1)), global_step=i+1) 
                        print ('model/dcgan-%d-%d saved' %(e+1, i+1))