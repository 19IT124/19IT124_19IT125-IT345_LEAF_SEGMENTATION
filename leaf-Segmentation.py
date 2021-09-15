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

