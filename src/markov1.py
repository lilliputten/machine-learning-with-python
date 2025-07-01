# Step 1: Initialize environment

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import tensorflow as tf

print('TensorFlow version:', tf.__version__)

# Step 2: Setting the distribution parameters using native TensorFlow
# Since TensorFlow Probability has compatibility issues, we'll implement HMM with native TF

# Initial distribution: 80% chance of cold (0), 20% chance of hot (1)
initial_probs = tf.constant([0.8, 0.2])

# Transition matrix: [cold->cold, cold->hot], [hot->cold, hot->hot]
# Cold day: 70% stay cold, 30% become hot
# Hot day: 20% become cold, 80% stay hot
transition_matrix = tf.constant([[0.7, 0.3], [0.2, 0.8]])

# Observation parameters: [cold_mean, hot_mean], [cold_std, hot_std]
obs_means = tf.constant([0.0, 15.0])  # Cold: 0°, Hot: 15°
obs_stds = tf.constant([5.0, 10.0])   # Cold: ±5°, Hot: ±10°

print('Initial probabilities:', initial_probs.numpy())
print('Transition matrix:', transition_matrix.numpy())
print('Observation means:', obs_means.numpy())
print('Observation stds:', obs_stds.numpy())

# Step 3: Predict expected temperatures for 7 days
def predict_temperatures(num_days=7):
    """Predict expected temperatures using forward algorithm"""
    # Start with initial distribution
    state_probs = initial_probs
    expected_temps = []

    for day in range(num_days):
        # Calculate expected temperature for current day
        expected_temp = tf.reduce_sum(state_probs * obs_means)
        expected_temps.append(expected_temp.numpy())

        # Update state probabilities for next day
        state_probs = tf.linalg.matvec(transition_matrix, state_probs, transpose_a=True)

    return expected_temps


# Predict temperatures for a week
temperatures = predict_temperatures(7)
print('\nExpected temperatures for 7 days:')
for i, temp in enumerate(temperatures, 1):
    print(f'Day {i}: {temp:.2f}°C')
