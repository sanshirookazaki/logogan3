#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import os
import time
from glob import glob
import tensorflow as tf

from ops import *
from utils import *

class DCGAN(object):
    def __init__(self, sess, image_size=128, is_crop=True,     #image_size関係なし
                 batch_size=64, sample_size = 64, image_shape=[128, 128, 3],  # 64
                 y_dim=None, z_dim=40, gf_dim=128, df_dim=128,   # none, 100 , 64, 64
                 gfc_dim=128, dfc_dim=128, c_dim=3, dataset_name='default', # 1024 , 1024 , 3
                 checkpoint_dir=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop                  
        self.batch_size = batch_size            
        self.image_size = image_size            
        self.sample_size = sample_size          
        self.image_shape = image_shape          

        self.y_dim = y_dim                      
        self.z_dim = z_dim                      

        self.gf_dim = gfc_dim                   
        self.df_dim = dfc_dim                   

        self.gfc_dim = gfc_dim                  
        self.dfc_dim = dfc_dim                  

        self.c_dim = 3  



        self.dataset_name = dataset_name    # celebA
        self.checkpoint_dir = checkpoint_dir  # checkpoint
        self.build_model()                      

    def build_model(self):
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape,      # [64 ,64,64,3] real_image
                                    name='real_images')
        self.sample_images = tf.placeholder(tf.float32, [self.sample_size] + self.image_shape,  # [64 ,64,64,3] sample_image
                                        name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim],                             # [None , 100]
                                name='z')

        #乱数から画像生成
        self.G  = self.generator(self.z)               
        # 学習データを入れた判別
        # dicriminator # sig[64 , -1] , [64 -1] 判別予測 , 特徴テンソル                                      #tanh(h4) shape_[64,64,64,3]
        self.D, self.D_logits  = self.discriminator(self.images)   
        
        #サンプル画像の生成　
        self.sampler  = self.sampler(self.z)                                                 # zのdeconv
        # dicriminator # sig[64 , -1] , [64 -1] 判別予測 , 特徴テンソル
        # generatorで生成した画像を入れた判別
        self.D_, self.D_logits_  = self.discriminator(self.G, reuse=True)      #使い回し             
        
        #生成された画像の平均値と学習データ画像の平均値
        #self.feature_matching = 0.1
        #self.mean_from_g = tf.reduce_mean(self.G )#, reduction_indices=(0))      #生成画像平均値
        #self.mean_from_i = tf.reduce_mean(self.images)# , reduction_indices=(0)) #学習画像平均値
        #平均値の差分に適度な値をかける
        #self.mean_loss = tf.mul(tf.nn.l2_loss(self.mean_from_g - self.mean_from_i), self.feature_matching)

        #判別器の特徴(生成画像/学習画像)の平均値
        #self.feature_form_i = tf.reduce_mean( self.D_feature )#, reduction_indices=(0))
        #self.feature_from_g = tf.reduce_mean( self.D_feature_ )#, reduction_indices=(0))
        #平均値の差分に適度な値をかける
        #self.feature_loss =  tf.mul(tf.nn.l2_loss(self.feature_from_g - self.feature_form_i), self.feature_matching)
        #self.losses = self.mean_loss + self.feature_loss 
        #tf.add_to_collection('ge_loss',self.losses)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        #tf.add_to_collection('ge_loss',self.ge_loss)
                                                    
        self.d_loss = self.d_loss_real + self.d_loss_fake    #disc _loss合計
        

        
        t_vars = tf.trainable_variables()       # return list all variables created with trainable=True.

        self.d_vars = [var for var in t_vars if 'd_' in var.name]   # d_ リスト
        self.g_vars = [var for var in t_vars if 'g_' in var.name]   # g_　リスト


        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""
        print "train()"
        with tf.device('/gpu:1'):     #1080SLIなし
	        data = glob(os.path.join("./data", config.dataset, "*.jpg"))    # ./data/celebA/*.jpg => jpgの画像リスト[image1.jpg,image2.jpg,...]
	        #np.random.shuffle(data)
	        print self.g_loss
	        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
	                          .minimize(self.d_loss, var_list=self.d_vars)                  # d_lossの最適化

	        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
	                          .minimize(self.g_loss, var_list=self.g_vars)                  # g_lossの最適化

	        tf.initialize_all_variables().run()

	        #self.saver = tf.train.Saver()
	        
	        #self.writer = tf.train.SummaryWriter("./logs", self.sess.graph_def)
	        #summary_op = tf.merge_all_summaries()
	        #self.summary_writer = tf.train.SummaryWriter('logs', graph_def=self.sess.graph_def)

	        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim)) # -1 ~ 1 size = (64, 100)
	        sample_files = data[0:self.sample_size] # data[0~64] 実画像
	        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files] # list image as array　sample= cropped_image 
	        sample_images = np.array(sample).astype(np.float32)  # cast image to array
	        
	        #counter = 1
	        start_time = time.time()

	        if self.load(self.checkpoint_dir): 
	            print(" [*] Load SUCCESS")
	        else:
	            print(" [!] Load failed...")


	        for epoch in xrange(config.epoch):          # epoch = 25
	            print epoch,"epoch"
	            data = glob(os.path.join("./data", config.dataset, "*.jpg")) # ./data/celebA/*.jpg => jpgの画像リスト
	            batch_idxs = min(len(data), config.train_size)/config.batch_size   # batch_idxs = len(data) / 64

	            for idx in xrange(0, batch_idxs):   # len(data)/64
	                print idx
	                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]  
	                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]   # image => array  (ops.py :scipy)
	                print len(batch)
	                batch_images = np.array(batch).astype(np.float32)

	                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32) #np.float32
	                                                                         # -1 ~ 1 [64,100]

	                # Update D network
	                summary_str = self.sess.run([d_optim],               
	                    feed_dict={ self.images: batch_images, self.z: batch_z })       
	                
	                print "D"
	                
	           
	                # Update G network
	                summary_str = self.sess.run([g_optim],
	                    feed_dict={ self.z: batch_z })  #   feature_matching >>> self.images: batch
	                
	                print "G"
	                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
	                summary_str = self.sess.run([g_optim],
	                    feed_dict={ self.z: batch_z })
	                
	                print "clear"        
	                #counter += 1
	                # 1ループあたりの時間を表示
	                print("Epoch: [%2d] [%4d/%4d] time: %4.4f") % (epoch, idx, batch_idxs,
	                        time.time() - start_time)

	                
	                #checkpointをsave
	                #if np.mod(counter, 500) == 2:
	                #    self.save(config.checkpoint_dir, counter)
	                
	                # 生成した画像を保存する　        
	                if epoch % 10 == 0 and idx % 15 == 0 :    #10epochごとのbatch15ごとに生成
	                    #画像保存
	                    z_sample = np.random.uniform(-0.5, 0.5, size=(self.batch_size, self.z_dim))  
	                    samples = self.sess.run(self.sampler, feed_dict={self.z: z_sample})
	                    save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	                    print "save image"
    def discriminator(self, image, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()  #使い回し可能

        
        if not self.y_dim:
            h0 = lrelu(conv2d(image, self.df_dim*1, name='d_h0_conv'))            
            h1 = lrelu(conv2d(h0, self.df_dim*2, name='d_h1_conv')) # 128
            h2 = lrelu(conv2d(h1, self.df_dim*4, name='d_h2_conv')) # 256
            h3 = lrelu(conv2d(h2, self.df_dim*8, name='d_h3_conv')) # 512
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')   # [64 , -1]
            #print h4

            return tf.nn.sigmoid(h4), h4 
            # sig[64 , -1] , [64 -1]=予測判別テンソル , 判別の特徴


    def generator(self, z, y=None):
        if not self.y_dim:
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*8*8, 'g_h0_lin', with_w=True)     
                                                                                                    
            self.h0 = tf.reshape(self.z_, [-1, 8, 8, self.gf_dim * 8]) 
            mean0, var0 = tf.nn.moments(self.h0, [0,1,2])                             
            h0 = tf.nn.relu(tf.nn.batch_normalization(self.h0, mean0, var0, None , None,1e-5,name='bn0'))     
            print "#",h0

            self.h1, self.h1_w, self.h1_b = deconv2d(h0,                                                                                
                [self.batch_size, 16, 16, self.gf_dim*4], name='g_h1', with_w=True)       
            mean1, var1 = tf.nn.moments(self.h1, [0,1,2])
            h1 = tf.nn.relu(tf.nn.batch_normalization(self.h1, mean1, var1, None , None,1e-5,name='bn1'))       
            print "#",h1

            h2, self.h2_w, self.h2_b = deconv2d(h1,                                     
                [self.batch_size, 32, 32, self.gf_dim*2], name='g_h2', with_w=True)     
            mean2, var2 = tf.nn.moments(h2, [0,1,2])
            h2 = tf.nn.relu(tf.nn.batch_normalization(h2, mean2, var2, None , None,1e-5,name='bn2'))                         
            print "#",h2

            h3, self.h3_w, self.h3_b = deconv2d(h2,
                [self.batch_size, 64, 64, self.gf_dim*1], name='g_h3', with_w=True)
            mean3, var3 = tf.nn.moments(h3, [0,1,2])
            h3 = tf.nn.relu(tf.nn.batch_normalization(h3, mean3, var3, None , None,1e-5,name='bn3'))
            print "#",h3

            h4, self.h4_w, self.h4_b = deconv2d(h3,
                [self.batch_size, 128, 128, 3], name='g_h4', with_w=True)                 
             
            return tf.nn.tanh(h4)                                                    
       
    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            # project `z` and reshape                                     
            h0 = tf.reshape(linear(z, self.gf_dim*8*8*8, 'g_h0_lin'),      
                            [-1, 8, 8, self.gf_dim * 8])
            mean0, var0 = tf.nn.moments(self.h0, [0,1,2])
            h0 = tf.nn.relu(tf.nn.batch_normalization(h0, mean0, var0, None , None,1e-5,name='bn0'))                      

            h1 = deconv2d(h0, [self.batch_size, 16, 16, self.gf_dim*4], name='g_h1') 
            mean1, var1 = tf.nn.moments(self.h1, [0,1,2])
            h1 = tf.nn.relu(tf.nn.batch_normalization(h1, mean1, var1, None , None,1e-5,name='bn1'))  
                             # batch_norm

            h2 = deconv2d(h1, [self.batch_size, 32, 32, self.gf_dim*2], name='g_h2') 
            mean2, var2 = tf.nn.moments(h2, [0,1,2])
            h2 = tf.nn.relu(tf.nn.batch_normalization(h2, mean2, var2, None , None,1e-5,name='bn2'))    

            h3 = deconv2d(h2, [self.batch_size, 64, 64, self.gf_dim*1], name='g_h3')  
            mean3, var3 = tf.nn.moments(h3, [0,1,2])
            h3 = tf.nn.relu(tf.nn.batch_normalization(h3, mean3, var3, None , None,1e-5,name='bn3'))

            h4 = deconv2d(h3, [self.batch_size, 128, 128, 3], name='g_h4')    

            return tf.nn.tanh(h4)   
        

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)      
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)        

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)             
        if ckpt and ckpt.model_checkpoint_path:                         
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)        
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))  
            return True
        else:
            return False