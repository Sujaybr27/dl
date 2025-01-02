import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.datasets import imdb
vocab_size = 10000
max_length = 100
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
train_padded = pad_sequences(train_data, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_data, maxlen=max_length, padding='post', truncating='post')
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length), # Embedding layer
    LSTM(64), 
    Dropout(0.5), 
    Dense(1, activation='sigmoid') 
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels))
