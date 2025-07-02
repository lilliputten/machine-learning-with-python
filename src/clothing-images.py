# Step 1: Imports

# TensorFlow and tf.keras
import matplotlib.pyplot as plt

# Helper libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Step 2: Dataset

fashion_mnist = keras.datasets.fashion_mnist  # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into tetsing and training

# Debug
train_images.shape
# > (60000, 28, 28)
train_images[0, 23, 23]  # let's have a look at one pixel
# > np.uint8(194)
train_labels[:10]  # let's have a look at the first 10 training labels
# > array([9, 0, 0, 3, 0, 2, 7, 2, 5, 5], dtype=uint8)

# Step 3: Create names for labels

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Step 4: Show an image

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

# Step 5: Data Preprocessing

train_images = train_images / 255.0

test_images = test_images / 255.0

# Step 6: Building the Model

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
        keras.layers.Dense(128, activation="relu"),  # hidden layer (2)
        keras.layers.Dense(10, activation="softmax"),  # output layer (3)
    ]
)

# Step 7: Compile the Model

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Step 8: Training the Model

model.fit(train_images, train_labels, epochs=10)  # we pass the data, labels and epochs and watch the magic!

# Epoch 1/10
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 3ms/step - accuracy: 0.7781 - loss: 0.6278
# ...
# Epoch 10/10
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 4ms/step - accuracy: 0.9111 - loss: 0.2409

# Step 8: Evaluating the Model

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

print("Test accuracy:", test_acc)

# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8875 - loss: 0.3373
# Test accuracy: 0.8851000070571899

# Step 9: Making Predictions

predictions = model.predict(test_images)

# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step

# Debug
predictions[0]

# array([4.3684032e-11, 4.7063287e-11, 1.1327360e-14, 7.2166453e-12,
#        4.0533611e-13, 7.0521501e-06, 1.2342988e-10, 1.2951012e-03,
#        3.7609055e-10, 9.9869788e-01], dtype=float32)

np.argmax(predictions[0])

# np.int64(9)

test_labels[0]

# np.uint8(9)

# Step 10: Verifying Predictions

COLOR = "white"
plt.rcParams["text.color"] = COLOR
plt.rcParams["axes.labelcolor"] = COLOR


def predict(model, image, correct_label):
    class_names = [
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
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]

    show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Excpected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def get_number():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return int(num)
        else:
            print("Try again...")


num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
