# Step 1: Imports

from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt

# Helper libraries
import numpy as np

# TensorFlow and tf.keras
import tensorflow as tf

keras = tf.keras


# Step 2: Dataset

fashion_mnist = keras.datasets.fashion_mnist  # Load sample dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # Split into tetsing and training

# Debug
print("Show train data shape")
print("train_images.shape:", train_images.shape)
# > (60000, 28, 28) -> 60K 28x28 bitmaps
print("train_images[0, 23, 23]:", train_images[0, 23, 23])  # Let's have a look at one pixel
# > np.uint8(194)
print("train_labels[:10]:", train_labels[:10])  # Let's have a look at the first 10 training labels
# > array([9, 0, 0, 3, 0, 2, 7, 2, 5, 5], dtype=uint8)


# Step 3: Create names for labels

class_names = [
    # Clothing class names
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# Step 4: DEBUG: Show a train image

num = 5
image = train_images[num]
label = train_labels[num]
class_name = class_names[label]
print(f"Show the image #{num} (class #{label}: {class_name})")

plt.figure()
plt.title(f"Train image: {class_name}")
plt.xlabel(f"Image #{num}, label #{label}")
plt.imshow(image)
plt.colorbar()
plt.show()


# Step 5: Data Preprocessing

# Convert (0:255) -> (0:1)
train_images = train_images / 255.0
test_images = test_images / 255.0


# Step 6: Building the Model

model: tf.keras.models.Sequential = keras.Sequential(
    [
        keras.layers.Input(shape=(28, 28)),  # Input layer (1)
        keras.layers.Flatten(),  # Flatten layer (2)
        keras.layers.Dense(128, activation="relu"),  # Hidden layer (3)
        keras.layers.Dense(10, activation="softmax"),  # Output layer (4)
    ]
)


# Step 7: Compile the Model

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# Step 8: Training the Model

do_epochs = 2
model.fit(train_images, train_labels, epochs=do_epochs)  # We pass the data, labels and epochs and watch the magic!

# Epoch 1/10
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 3ms/step - accuracy: 0.7781 - loss: 0.6278
# ...
# Epoch 10/10
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 4ms/step - accuracy: 0.9111 - loss: 0.2409


# Step 8: Evaluating the Model

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8392 - loss: 0.4216
# Test loss: 0.42236772179603577
# Test accuracy: 0.8398000001907349


# Step 9: Making Predictions

predictions = model.predict(test_images)

# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step

# Debug
print("predictions[0]:", predictions[0])

# array([4.3684032e-11, 4.7063287e-11, 1.1327360e-14, 7.2166453e-12,
#        4.0533611e-13, 7.0521501e-06, 1.2342988e-10, 1.2951012e-03,
#        3.7609055e-10, 9.9869788e-01], dtype=float32)

print("np.argmax(predictions[0]):", np.argmax(predictions[0]))

# np.int64(9)

print("test_labels[0]:", test_labels[0])

# np.uint8(9)


# Step 10: Verifying Predictions


def get_number():
    while True:
        num = input("Pick a test image number (0-9999): ")
        if num.isdigit():
            num = int(num)
            if 0 <= num < 10000:
                return int(num)
        else:
            print("Try again...")


def show_image(image, real_class, predicted_class):
    plt.figure()
    plt.title(f"Predicted: {predicted_class}")
    plt.xlabel(f"Real: {real_class}")
    plt.imshow(image)
    plt.colorbar()
    plt.show()


def predict(model, image, real_label):
    """
    Predict a test image class by the image
    """
    image_array = np.array([image])

    # Get prediction indices list for an image
    prediction = model.predict(image_array)

    real_class = class_names[real_label]

    predicted_label = np.argmax(prediction)
    predicted_class = class_names[predicted_label]

    print("real_label:", real_label)
    print("prediction:", prediction)
    print("predicted_label:", predicted_label)
    print("predicted_class:", predicted_class)
    print("real_class:", real_class)

    print("Show the predicted image")
    show_image(image, real_class, predicted_class)


# Try to predict a test image class by the image
num = 5  # get_number()  # Get the number interactively
print("Using test image:", num)
image = test_images[num]
real_label = test_labels[num]

predict(model, image, real_label)

print("Done")
