import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (150, 150)

model = tf.keras.models.load_model("model.h5")

def predict(image_path):
    img = Image.open(image_path).resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        print("Dog 🐶")
    else:
        print("Cat 🐱")

predict("test.jpg")