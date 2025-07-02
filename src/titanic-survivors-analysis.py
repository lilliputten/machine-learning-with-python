from __future__ import absolute_import, division, print_function, unicode_literals

from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

print("Start")

# Load dataset.
dftrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")  # training data
dfeval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv")  # testing data
y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")

print("Done")

print("Head:\n", dftrain.head())
print("Desribe:\n", dftrain.describe())
print("Shape:\n", dftrain.shape)
print("Ages:\n", dftrain.age.hist(bins=20))
