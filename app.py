import os

import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory,redirect
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import json
import urllib.request

app = Flask(__name__)
classes = ["Bathroom Mirrors", "Bathroom Shelves", "Bathroom Sinks", "Ceiling Fans", "Curtain Rods", "Pet Beds", "Rugs", "Toilet Paper Holders", "Towel Bars", "Towel Hooks", "Towel Racks", "Towel Rings"]
transfer_learning_model = tf.keras.models.Sequential()
is_model_trained = 'false'

# home page
@app.route("/")
def home():
    return render_template("test.html")

def loadImages():
  with urllib.request.urlopen('https://gist.github.com/shibicr93/6d5550dea9252af511a18a5a8264bf50/raw/fa77f4454e7e9db7186dc4dbb9b3ac207f7a5f78/imgurls_12*100.json') as url:
      array = json.loads(url.read().decode())

  for item in array:
    for key, values in item.items():
      for value in values: 
        try:
            train_dir = "./train/"+key
            url = value
            if not os.path.exists(train_dir):
                  os.makedirs(train_dir)
            filename = url.split('/')[-1]
            filepath = train_dir+"/"+filename
            if not os.path.exists(filepath):
              urllib.request.urlretrieve(url, filepath)
        except Exception as e:
                    continue

@app.route("/ismodeltrained")
def isModelTrained():
    return is_model_trained, 200                    

@app.route("/train")
def trainModel():
  loadImages()
  vgg_model = VGG16(input_shape=[100,100,3], weights='imagenet', include_top=False)
  for layer in vgg_model.layers:
    layer.trainable = False


  TRAINING_DIR = "./train/"
  training_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

  train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(100,100),
    batch_size=64,
    class_mode='categorical',
    subset='training'
  )
  
  transfer_learning_model.add(vgg_model)

  transfer_learning_model.add(Conv2D(1024, kernel_size=3, padding='same'))

  transfer_learning_model.add(Activation('relu'))

  transfer_learning_model.add(MaxPooling2D(pool_size=(2, 2)))
  transfer_learning_model.add(Dropout(0.3))

  transfer_learning_model.add(Flatten())
  transfer_learning_model.add(Dense(150))
  transfer_learning_model.add(Activation('relu'))
  transfer_learning_model.add(Dropout(0.4))
  transfer_learning_model.add(Dense(12,activation = 'softmax'))
  transfer_learning_model.summary()
  transfer_learning_model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  history = transfer_learning_model.fit(train_generator, epochs=30, verbose = 2)
  is_model_trained = true


@app.route('/predict',methods=['POST'])
def predict():
    file = request.files['imgfile']

    imgpath = os.path.join(app.root_path, file.filename)
    file.save(imgpath)
    preprocessed_image=tf.io.read_file(imgpath)
    preprocessed_image=tf.image.decode_jpeg(preprocessed_image,channels=3)
    preprocessed_image=tf.image.resize(preprocessed_image,[100,100])
    preprocessed_image/=255.0
    preprocessed_image=tf.reshape(preprocessed_image,(1,100,100,3));
    res = transfer_learning_model.predict_classes(preprocessed_image, 1, verbose=0)[0]
    print("response:",classes[res])
    print(imgpath," Image saved successfully")
    return "https://www.lowes.com/search?searchTerm="+classes[res], 200

if(__name__=='__main__'):
    app.run(port=8080,debug='true',threaded=False)

