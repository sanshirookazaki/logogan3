#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 5000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 128, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)
                                                    # checkpoint
    if not os.path.exists(FLAGS.checkpoint_dir): #ディレクトリがなければ
        os.makedirs(FLAGS.checkpoint_dir)       #ディレクトリ作成
    if not os.path.exists(FLAGS.sample_dir):    #samples
        os.makedirs(FLAGS.sample_dir)
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True    
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:   # config=config
        if FLAGS.dataset == 'mnist':     # dataset default"celebA"   == "mnist"の場合
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, y_dim=10,
                    dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)
        else:
            #celebA
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,                 #FLAGS.image_size = 108 , batch_size = 64      
                    dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir) #dataset = "celebA" , is_crop = False , checkpoint_dir = "checkpoint"

        if FLAGS.is_train:
    
            dcgan.train(FLAGS)  
        else:
            dcgan.load(FLAGS.checkpoint_dir)    # モデル復元


        

if __name__ == '__main__':
    tf.app.run()

