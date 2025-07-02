# Step 1: Initialize environment

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Step 2: Setting the distribution parameters using native TensorFlow Hidden Markov Models

# Since TensorFlow Probability has compatibility issues, we'll implement HMM with native TF

# Initial distribution: 80% chance of cold (0), 20% chance of hot (1)
initial_probs = tf.constant([0.8, 0.2])

# Transition matrix: [cold->cold, cold->hot], [hot->cold, hot->hot], former `tfd.Categorical` creation
transition_matrix = tf.constant(
    [
        # Cold day: 70% stay cold, 30% become hot
        [0.7, 0.3],
        # Hot day: 20% become cold, 80% stay hot
        [0.2, 0.8],
    ]
)

# Observation parameters: [cold_mean, hot_mean], [cold_std, hot_std]
obs_means = tf.constant([0.0, 15.0])  # Cold: 0°, Hot: 15°, former `loc` parameter for `tfd.Normal`
obs_stds = tf.constant(
    [5.0, 10.0]
)  # Cold: ±5°, Hot: ±10°, former `scale` parameter for `tfd.Normal`, not used actually

print("Initial probabilities:", initial_probs.numpy())
print("Transition matrix:", transition_matrix.numpy())
print("Observation means:", obs_means.numpy())
print("Observation stds:", obs_stds.numpy())

# Step 2a: Sample actual temperatures using standard deviations


def sample_temperatures(num_days=7):
    """Sample actual temperatures from normal distributions"""
    state_probs = initial_probs
    sampled_temps = []

    for day in range(num_days):
        # Sample from normal distributions for each state
        cold_sample = tf.random.normal([1], obs_means[0], obs_stds[0])
        hot_sample = tf.random.normal([1], obs_means[1], obs_stds[1])

        # Weighted average based on state probabilities
        temp_sample = state_probs[0] * cold_sample + state_probs[1] * hot_sample
        sampled_temps.append(temp_sample.numpy()[0])

        # Update state probabilities
        state_probs = tf.linalg.matvec(transition_matrix, state_probs, transpose_a=True)

    return sampled_temps


print("\nSampled temperatures (random, using std deviations):")
sampled = sample_temperatures(7)
for i, temp in enumerate(sampled, 1):
    print(f"Day {i}: {temp:.2f}°C")


# Step 3: Predict expected temperatures using proper HMM mean calculation


def predict_temperatures_hmm(num_days=7):
    """Calculate HMM mean like original TFP model.mean()"""
    # This matches the original tfd.HiddenMarkovModel behavior
    # where observation_distribution contributes to the overall mean

    state_probs = initial_probs
    expected_temps = []

    for day in range(num_days):
        # Expected observation = sum over states of (state_prob * obs_mean)
        # This is exactly what the original observation_distribution did
        expected_temp = tf.reduce_sum(state_probs * obs_means)
        expected_temps.append(expected_temp.numpy())

        # Forward step: update state probabilities
        state_probs = tf.linalg.matvec(transition_matrix, state_probs, transpose_a=True)

    return expected_temps


# Predict temperatures for a week (equivalent to original model.mean())
temperatures = predict_temperatures_hmm(7)
print("\nPredicted temperatures (HMM mean, equivalent to original TFP):")
for i, temp in enumerate(temperatures, 1):
    print(f"Day {i}: {temp:.2f}°C")

# print("\nAs array (like original output):", temperatures)

# We expect to get this output:
#
# Day 1: 3.00°C
# Day 2: 6.00°C
# Day 3: 7.50°C
# Day 4: 8.25°C
# Day 5: 8.63°C
# Day 6: 8.81°C
# Day 7: 8.91°C
#
# Compare with the expectactions in the original ipynb (but ensure the same input values):
#
# [3.        5.9999995 7.4999995 8.25      8.625001  8.812501  8.90625  ]
