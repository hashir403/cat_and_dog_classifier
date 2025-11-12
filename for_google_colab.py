# this code for google colab 


# Required setup for Kaggle datasets
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# Kaggle dataset link: https://www.kaggle.com/datasets/salader/dogs-vs-cats
!kaggle datasets download salader/dogs-vs-cats

# Unzip the downloaded file
import zipfile
zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
zip_ref.extractall()
zip_ref.close()

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
import matplotlib.pyplot as plt
import cv2
import shutil

# Load the dataset
train_ds = keras.utils.image_dataset_from_directory(
    directory="/content/train",
    labels="inferred",
    label_mode="int",  # Labels as integers (0 for cat, 1 for dog)
    batch_size=32,
    image_size=(256, 256),
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory="/content/test",
    labels="inferred",
    label_mode="int",  # Labels as integers
    batch_size=32,
    image_size=(256, 256),
)

# Normalization function (Corrected Code)
def process(image, label):
    # Normalize the image to [0, 1]
    image = tf.cast(image / 255.0, tf.float32)
    return image, label  # Keep the label as an integer

# Apply normalization to both datasets
train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# Create a CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding="valid", activation="relu", input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))

model.add(Conv2D(64, kernel_size=(3, 3), padding="valid", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))

model.add(Conv2D(128, kernel_size=(3, 3), padding="valid", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))  # Binary classification output
model.summary()

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
model.save("trained_model.h5")

print("Model saved as 'trained_model.h5'")

# Plot training and validation accuracy
plt.plot(history.history["accuracy"], color="red", label="train")
plt.plot(history.history["val_accuracy"], color="blue", label="validation")
plt.legend()
plt.title("Model Accuracy")
plt.show()

# Plot training and validation loss
plt.plot(history.history["loss"], color="red", label="train")
plt.plot(history.history["val_loss"], color="blue", label="validation")
plt.legend()
plt.title("Model Loss")
plt.show()

# Define a function to map label integers to string labels (for predictions)
def label_to_string(label):
    return 'cat' if label == 0 else 'dog'

# Test the model with an example image
test_img = cv2.imread("/content/cat1.jpg")
plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.show()

test_img = cv2.resize(test_img, (256, 256))
test_input = test_img.reshape((1, 256, 256, 3)) / 255.0  # Normalize the input
prediction = model.predict(test_input)
predicted_label = 1 if prediction > 0.5 else 0  # Binary threshold
print(f"Predicted: {label_to_string(predicted_label)}")

