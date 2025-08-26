import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

MODEL_FILE = "emnist_cnn.keras"

# -------------------------------
# Load EMNIST dataset (ByClass = 62 classes)
# -------------------------------
print(" Loading EMNIST dataset...")
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/byclass",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

NUM_CLASSES = ds_info.features["label"].num_classes

# Normalize images
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

ds_train = ds_train.map(normalize_img).batch(128).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(normalize_img).batch(128).prefetch(tf.data.AUTOTUNE)

# -------------------------------
# Build CNN model
# -------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# -------------------------------
# Train model
# -------------------------------
print(" Training model...")
model.fit(ds_train, epochs=5, validation_data=ds_test)

# -------------------------------
# Save model
# -------------------------------
model.save(MODEL_FILE)
print(f" Model saved as {MODEL_FILE}")
