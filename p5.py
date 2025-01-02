import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

IMG_SIZE, BATCH_SIZE = 150, 32

(train_data, test_data), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
)

def preprocess(image, label):
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
  return image, label

train_ds = train_data.map(preprocess).shuffle(1000).batch(BATCH_SIZE)
test_ds = test_data.map(preprocess).batch(BATCH_SIZE)

model = models.Sequential([
  layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_ds, epochs=10, validation_data=test_ds)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.2f}")
