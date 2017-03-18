#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import math
#import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True):  # [image1.jpg , ....] , image_size = 108 ,is_crop=
    return transform(imread(image_path), image_size, is_crop) # 

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)  # (images+1)/2 pathにセーブ

def imread(path):
    return scipy.misc.imread(path).astype(np.float) #image read as array
                                                    # ピクセルごとに[R G B]でリスト
                                                    ##    

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]  #height width 要素数
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=False):     # image_array 108 true
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image

        h ,w = image.shape[:2]

        cropped_image = scipy.misc.imresize(image[:h,:w],[128,128])

    return np.array(cropped_image)/127.5 - 1. # -1 ~ 1 にRGB正規

def inverse_transform(images):
    return (images+1.)/2.



def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]   # images[64/4]  16
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8) # 17 / 510 

  clip = mpy.VideoClip(make_frame, duration=duration) # 2 
  clip.write_gif(fname, fps = len(images) / duration) 