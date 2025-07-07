# Step 1: Initialiing

from dotenv import load_dotenv

load_dotenv()

import os
from typing import Any, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

keras = tf.keras

# Step 2: Load dataset using tf.keras.utils instead of tensorflow_datasets

# Original tensorflow-datasets based code (commented due to protobuf compatibility issues):
# import tensorflow_datasets as tfds
# tfds.disable_progress_bar()
# (raw_train, raw_validation, raw_test), metadata = tfds.load(
#     "cats_vs_dogs",
#     split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
#     with_info=True,
#     as_supervised=True,
# )
# get_label_name = metadata.features["label"].int2str
# for image, label in raw_train.take(5):
#   plt.figure()
#   plt.imshow(image)
#   plt.title(get_label_name(label))

print("Started data load...")

# Download and prepare the cats vs dogs dataset (using direct link instead of `tfds.load("cats_vs_dogs", ...)
url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
print("Data is already loaded")
path_to_zip = tf.keras.utils.get_file("cats_and_dogs.zip", origin=url, extract=True)
print("File is ready")
base_dir = os.path.dirname(path_to_zip)

# Check what directories exist
print(f"Base directory: {base_dir}")
print(f"Contents: {os.listdir(base_dir)}")

# The actual path is in the extracted directory
PATH = os.path.join(base_dir, "cats_and_dogs_extracted", "PetImages")
if not os.path.exists(PATH):
    # Try alternative paths
    extracted_dir = os.path.join(base_dir, "cats_and_dogs_extracted")
    if os.path.exists(extracted_dir):
        print(f"Extracted dir contents: {os.listdir(extracted_dir)}")
        for item in os.listdir(extracted_dir):
            item_path = os.path.join(extracted_dir, item)
            if os.path.isdir(item_path):
                try:
                    contents = os.listdir(item_path)
                    if "Cat" in contents and "Dog" in contents:
                        PATH = item_path
                        break
                except:
                    continue

# Check if the data path has been correctly located
print(f"Using PATH: {PATH}")
if os.path.exists(PATH):
    print(f"PATH contents: {os.listdir(PATH)}")
else:
    print("PATH does not exist")
    exit(1)


# Create datasets
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# Clean up corrupted images first
import glob

print("Cleaning corrupted images...")
for folder in ["Cat", "Dog"]:
    folder_path = os.path.join(PATH, folder)
    files_removed = 0
    for file_path in glob.glob(os.path.join(folder_path, "*")):
        try:
            img = tf.io.read_file(file_path)
            tf.image.decode_image(img)
        except:
            print(f"Removing corrupted file: {os.path.basename(file_path)}")
            os.remove(file_path)
            files_removed += 1
    print(f"Removed {files_removed} corrupted files from {folder}")

# Create training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# Create validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_dataset.class_names
print(f"Class names: {class_names}")

# # Step 3: Show some of the loaded images
#
# # Simple label function
# get_label_name: Callable[[Any], str] = lambda x: class_names[x] if x < len(class_names) else f"Label {x}"
#
# # Take a few samples for display
# raw_train = train_dataset.take(3)  # Take first batch for display
#
# # Display a few images from the dataset
# for batch_no, (images, labels) in enumerate(raw_train):
#     images_count = len(images)
#     show_images_count = min(2, len(images))
#     print(f"Show {show_images_count} of {images_count} images for the batch #{batch_no}...")
#     for i in range(show_images_count):
#         title = f"Batch #{batch_no}, image #{i}: {get_label_name(labels[i].numpy())}"
#         print(title)
#         plt.figure()
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(title)
#         plt.show()
#     print(f"Finished displaying of the batch #{batch_no}")
#     # break  # Only show first batch
#
# # NOTE: Set a breakpoint to show the displayed images
# print("Done")

# Step 4: Shuffle

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE)
validation_batches = validation_dataset
# test_batches = test_dataset.batch(BATCH_SIZE)

for img, label in train_dataset.take(2):
    print("New shape:", img.shape)

# Step 5: Picking a Pretrained Model

IMG_SHAPE = IMG_SIZE + (3,)  # IMG_SIZE is already (160, 160), so this becomes (160, 160, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")

base_model.summary()

# Step 6: Check the model

last_image = None

for image, _ in train_batches.take(1):
    last_image = image

feature_batch = base_model(last_image)
print(feature_batch.shape)

# Step 7: Freezing the Base

base_model.trainable = False
base_model.summary()

# Step 8: Adding our Classifier

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)
model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])
model.summary()

# Step 9: Training the Model

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# We can evaluate the model right now to see how it does before training it on our new images
initial_epochs = 3
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

# Now we can train it on our images
history = model.fit(train_batches, epochs=initial_epochs, validation_data=validation_batches)

acc = history.history["accuracy"]
print(acc)

model.save("dogs_vs_cats.h5")  # we can save the model and reload it at anytime in the future
new_model = tf.keras.models.load_model("dogs_vs_cats.h5")
