# Step 1: Initialiing

# %tensorflow_version 2.x  # this line is not required unless you are in a notebook

from dotenv import load_dotenv

load_dotenv()

import os
from typing import Any, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

keras = tf.keras

# Step 2: Load dataset using tf.keras.utils instead of tensorflow_datasets

# Original tensorflow-datasets code (commented due to protobuf compatibility issues):
# import tensorflow_datasets as tfds
# tfds.disable_progress_bar()
# (raw_train, raw_validation, raw_test), metadata = tfds.load(
#     'cats_vs_dogs',
#     split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
#     with_info=True,
#     as_supervised=True,
# )
# get_label_name = metadata.features['label'].int2str
# for image, label in raw_train.take(5):
#   plt.figure()
#   plt.imshow(image)
#   plt.title(get_label_name(label))

print("Started data load...")

# Download and prepare the cats vs dogs dataset (using direct link instead of `tfds.load('cats_vs_dogs', ...)
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

# Create training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    # @see https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
    PATH,
    validation_split=0.2, # Optional float between 0 and 1, fraction of data to reserve for validation.
    subset="training", # Subset of the data to return. One of "training", "validation", or "both". Only used if validation_split is set. When subset="both", the utility returns a tuple of two datasets (the training and validation datasets respectively).
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE, # Size of the batches of data. Defaults to 32. If None, the data will not be batched (the dataset will yield individual samples).
)

# Create validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    # @see https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
    PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_dataset.class_names
print(f"Class names: {class_names}")

# Step 3: Show some of the loaded images

# Simple label function
get_label_name: Callable[[Any], str] = lambda x: class_names[x] if x < len(class_names) else f"Label {x}"

# Take a few samples for display
raw_train = train_dataset.take(3)  # Take first batch for display

# Display a few images from the dataset
for batch_no, (images, labels) in enumerate(raw_train):
    images_count = len(images)
    show_images_count = min(2, len(images))
    print(f"Show {show_images_count} of {images_count} images for the batch #{batch_no}...")
    for i in range(show_images_count):
        title = f"Batch #{batch_no}, image #{i}: {get_label_name(labels[i].numpy())}"
        print(title)
        plt.figure()
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(title)
        plt.show()
    print(f"Finished displaying of the batch #{batch_no}")
    # break  # Only show first batch

# NOTE: Set a breakpoint to show the displayed images
print("Done")
