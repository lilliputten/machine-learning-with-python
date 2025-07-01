from __future__ import absolute_import, division, print_function, unicode_literals

# import tensorflow as tf
import keras
import numpy as np
import pandas as pd

# Step 1:

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Lets define some constants to help us later on

train_path = keras.utils.get_file(
    'iris_training.csv',
    'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv',
)
test_path = keras.utils.get_file(
    'iris_test.csv',
    'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv',
)

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe

train_y = train.pop('Species')
test_y = test.pop('Species')
print('Head:\n', train.head())

# Step 2:

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each using Keras
model = keras.Sequential(
    [
        # Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
        keras.layers.Dense(30, activation='relu', input_shape=(4,)),
        keras.layers.Dense(10, activation='relu'),
        # The model must choose between 3 classes.
        keras.layers.Dense(3, activation='softmax'),
    ]
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train.values, train_y.values, epochs=100, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test.values, test_y.values)
print(f'Test loass: {test_loss:.4f}')
print(f'Test accuracy: {test_accuracy:.4f}')

# Step 3:

import numpy as np

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}


def is_number(s):
    try:
        float(s)  # or int(s) for integers only
        return True
    except ValueError:
        return False


print('Please type numeric values as prompted.')
for feature in features:
    valid = False
    # val = None
    while True:
        val = input(feature + ': ')
        if is_number(val):
            predict[feature] = [float(val)]

    # if val is not None:
    #     predict[feature] = [float(val)]

# Convert input to numpy array for prediction
predict_x = np.array([[predict[feature][0] for feature in features]])

# Make prediction
predictions = model.predict(predict_x)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

print('\nPrediction is "{}" ({:.1f}%)'.format(SPECIES[predicted_class], 100 * confidence))
