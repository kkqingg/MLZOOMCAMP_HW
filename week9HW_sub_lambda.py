#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow import keras

model_beewasp = keras.models.load_model('model-bee-wasp.h5')

get_ipython().system('python -V')
tf.__version__

converter = tf.lite.TFLiteConverter.from_keras_model(model_beewasp)
tflite_model = converter.convert()

with open('model_beewasp.tflite','wb') as f_out:
    f_out.write(tflite_model)

#check whether model.tflite is in this same folder
get_ipython().system('dir')

pip install pillow


from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

url = 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'

img = download_image(url)
prep_img= prepare_image(img, (150,150))

#PREPROCESS THE IMAGE
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
x= np.array(prep_img)
X=np.array([x])
X = np.array(X, dtype=np.float32)
X=X/255.

import tensorflow.lite as tflite
from keras_image_helper import create_preprocessor

#Load the model
interpreter = tflite.Interpreter(model_path='model_beewasp.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


interpreter.set_tensor(input_index,X)
# interpreter.set_tensor(input_index, X.astype(np.uint8))
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


url2= 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'
