# Step 1: Initialiing

from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt

# %tensorflow_version 2.x  # this line is not required unless you are in a notebook
import tensorflow as tf

# Use aliases to avoid import resolution issues
datasets = tf.keras.datasets
layers = tf.keras.layers
models = tf.keras.models

# Step 2: Load & prepare data

#  LOAD AND SPLIT DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = [
    # Class names, respectively
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Step 3: Debug

# Let's look at a one image
IMG_INDEX = 7  # change this to look at other images

plt.imshow(train_images[IMG_INDEX])  # ,cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()

# Step 4: Create a model

model = models.Sequential()
model.add(layers.Input(shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))

# print("model.summary", model.summary())  # Let's have a look at our model so far

# Step 5: Adding layers

model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))

print("model.summary", model.summary())  # Show updated model

# Step 6: Training

model.compile(
    optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
)

do_epochs = 2
history = model.fit(train_images, train_labels, epochs=do_epochs, validation_data=(test_images, test_labels))

# Step 7: Evaluating the Model

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("test_acc:", test_acc)

# Step 8: Data augmentation

# Original deprecated code (for reference):
# from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator
# datagen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')
# test_img = train_images[20]
# img = image.img_to_array(test_img)
# img = img.reshape((1,) + img.shape)
# for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):
#     plt.figure(i)
#     plot = plt.imshow(image.img_to_array(batch[0]))
#     i += 1
#     if i > 4:
#         break

# Create data augmentation pipeline using modern tf.keras.layers
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ]
)

# Pick an image to transform and show augmented versions
test_img = train_images[20]
test_img_batch = tf.expand_dims(test_img, 0)  # Add batch dimension

# Show original and augmented images
for i in range(5):
    plt.figure(i)
    if i == 0:
        plt.imshow(test_img)
        plt.title("Original")
    else:
        augmented = data_augmentation(test_img_batch, training=True)
        plt.imshow(tf.squeeze(augmented, 0))
        plt.title(f"Augmented {i}")
    plt.axis("off")

plt.show()
