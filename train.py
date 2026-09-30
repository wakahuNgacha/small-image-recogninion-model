
import tensorflow as tf
from tensorflow.keras import layers, models

# Settings
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 5

# Load images from folders
train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42
)

class_names = train_data.class_names
print("Classes:", class_names)

# Improve data loading
train_data = train_data.prefetch(
    tf.data.AUTOTUNE
)
validation_data = validation_data.prefetch(
    tf.data.AUTOTUNE
)

# Load pretrained lightweight model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# Build classifier
model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation="softmax")
])

# Configure training
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

# Save model and class names
model.save("model/image_model.keras")

with open("model/classes.txt", "w") as f:
    f.write("\n".join(class_names))

print("Model saved successfully!")
