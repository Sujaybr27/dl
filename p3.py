# -*- coding: utf-8 -*-
"""p3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rSzoDcZDmA2hYDBKmlafMukLIL9diQ8t
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical

# Load data and prepare word index
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
word_index = reuters.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Decode a sample newswire (optional)
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# Vectorize data
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train, x_test = vectorize_sequences(train_data), vectorize_sequences(test_data)
one_hot_train_labels, one_hot_test_labels = to_categorical(train_labels), to_categorical(test_labels)

# Build and compile model
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(46, activation='softmax')
])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Split validation data
x_val, partial_x_train = x_train[:1000], x_train[1000:]
y_val, partial_y_train = one_hot_train_labels[:1000], one_hot_train_labels[1000:]

# Train model
history = model.fit(
    partial_x_train, partial_y_train,
    epochs=25, batch_size=512, validation_data=(x_val, y_val)
)

# Evaluate model
results = model.evaluate(x_test, one_hot_test_labels)