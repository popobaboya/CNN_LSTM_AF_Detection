import numpy as np
import os

import tensorflow as tf
assert tf.__version__.startswith('2') #tensorflow 2번째꺼로 시작

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader


import matplotlib.pyplot as plt


image_path = tf.keras.utils.get_file(
      'flower_photos',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      untar=True)

# print("image path = ", image_path)

data = DataLoader.from_folder(image_path) #image path로 부터 data를 받아와서 data 변수에 저장
train_data, test_data = data.split(0.9) # train data와 test data를 9대 1로 나눈다. 

# 2단계. TensorFlow 모델을 사용자 지정합니다.
model = image_classifier.create(train_data, epochs=4) #모델의 서머리를 볼수있는 것이 create라는 method에 포함되어 있다.


# 3단계. 모델을 평가합니다.
loss, accuracy = model.evaluate(test_data)