
import tensorflow as tf
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python predict.py image.jpg")
    sys.exit(1)

# Load model and class names
model = tf.keras.models.load_model(
    "model/image_model.keras"
)

with open("model/classes.txt") as f:
    classes = f.read().splitlines()

# Load and prepare image
image = tf.keras.utils.load_img(
    sys.argv[1],
    target_size=(160, 160)
)

image = tf.keras.utils.img_to_array(image)
image = np.expand_dims(image, axis=0)

# Predict
predictions = model.predict(image, verbose=0)[0]
index = np.argmax(predictions)

print("Prediction:", classes[index])
print("Confidence:", round(float(predictions[index]) * 100, 2), "%")