"""## Import packages"""

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from tensorflow.keras.layers import  Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import glob
import random
import cv2
from random import shuffle

def image_generator(files, batch_size =10 , sz = (256, 256)):
  
  while True: 
    
    #extract a random batch 
    batch = np.random.choice(files, size = batch_size)    
    
    #variables for collecting batches of inputs and outputs 
    batch_x = []
    batch_y = []
    
    
    for f in batch:
        
        #get the masks. Note that masks are png files 
        mask = Image.open(f[-9::-1][::-1]+'_fg.png')
        mask = np.array(mask.resize(sz))


        #preprocess the mask 
        mask[mask >= 2] = 0 
        mask[mask != 0 ] = 1
        
        batch_y.append(mask)

        #preprocess the raw images 
        raw = Image.open(f)
        raw = raw.resize(sz)
        raw = np.array(raw)

        #check the number of channels because some of the images are RGBA or GRAY
        if len(raw.shape) == 2:
          raw = np.stack((raw,)*3, axis=-1)

        else:
          raw = raw[:,:,0:3]

        batch_x.append(raw)

    #preprocess a batch of images and masks 
    batch_x = np.array(batch_x)/255.
    batch_y = np.array(batch_y)
    batch_y = np.expand_dims(batch_y,3)

    yield (batch_x, batch_y)

all_files=[]
for i in os.listdir('/content/gdrive/MyDrive/trainingdata/A1'):
  if i[-5:]=='b.png':
    all_files.append('/content/gdrive/MyDrive/trainingdata/A1/'+i)  
for i in os.listdir('/content/gdrive/MyDrive/trainingdata/A2'):
  if i[-5:]=='b.png':
    all_files.append('/content/gdrive/MyDrive/trainingdata/A2/'+i)   
for i in os.listdir('/content/gdrive/MyDrive/trainingdata/A3'):
  if i[-5:]=='b.png':
    all_files.append('/content/gdrive/MyDrive/trainingdata/A3'+i) 
#for i in os.listdir('/content/gdrive/My Drive/training/images/A4'):
#  if i[-5:]=='b.png':
#    all_files.append('/content/gdrive/My Drive/training/images/A4/'+i)

batch_size = 10
shuffle(all_files)
split = int(0.9 * len(all_files))
#split into training and testing
train_files = all_files[0:split]
test_files  = all_files[split:]

train_generator = image_generator(train_files, batch_size = batch_size)
test_generator  = image_generator(test_files, batch_size = batch_size)

x, y= next(train_generator)

def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou
  
def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou

"""# Training"""

train_steps = len(train_files) //batch_size
test_steps = len(test_files) //batch_size
#model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = [mean_iou])
model.fit_generator(train_generator, 
                    epochs = 100, steps_per_epoch = train_steps,validation_data = test_generator, validation_steps = 10,verbose = 2)

model.save_weights('/content/gdrive/My Drive/training/weights.h5')
