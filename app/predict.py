import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
#from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('./model/SkinCancer_model_v5.h5')

def preprocess_img(img):
  #img = img_to_array(img)
  img = np.expand_dims(img,0)
  img = preprocess_input(img)
  return img

def make_prediction(file):
  img       = np.array(Image.open(file).resize((224,224)),dtype=np.float32)
  #img        = tf.keras.utils.load_img(file,target_size=(224,224,3))
  tensor     = preprocess_img(img)
  prediction = model.predict(tensor)
  return np.argmax(prediction)

