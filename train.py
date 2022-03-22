import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# %% codecell
# general parameters
run = "example"
version = 0
model_num = 9
slice_size = 128

num_epochs = 5
batch_size = 32
split = 0.8

# general functions
def read_image(im1_file, im2_file):
    im1_image = tf.io.read_file(directory + im1_file)
    im1_image = tf.image.decode_image(im1_image, dtype = tf.float32)
    im1_image.set_shape([slice_size, slice_size, 1])

    im2_image = tf.io.read_file(directory + im2_file)
    im2_image = tf.image.decode_image(im2_image, dtype = tf.float32)
    im2_image.set_shape([slice_size, slice_size, 1])

    return im1_image, im2_image

# %% codecell
# initialize dataset
directory = 'datasets/' + run + '_v' + str(version) + '/'
df = pd.read_csv('datasets/' + run + '_v' + str(version) + '.csv')

gds_paths = df['GDS'].values
sem_paths = df['SEMb'].values

ds = tf.data.Dataset.from_tensor_slices((gds_paths, sem_paths))

DATASET_SIZE = len(gds_paths)
ds = ds.shuffle(buffer_size = len(df), reshuffle_each_iteration = False)
train_size = int(split*DATASET_SIZE)
train_dataset = ds.take(train_size)
test_dataset = ds.skip(train_size)

train_dataset = train_dataset.shuffle(buffer_size = len(train_dataset),
    reshuffle_each_iteration = True)

ds_train = train_dataset.map(read_image).batch(batch_size)
ds_test = test_dataset.map(read_image).batch(batch_size)

print('Size of ds_train: ' + str(len(ds_train)))

# %% codecell
# construct the model
model = models.Sequential()
model.add(layers.Conv2D(1, (3, 3), activation = 'relu',  padding = 'same',
    input_shape = (slice_size, slice_size, 1)))
model.add(layers.AveragePooling2D((2, 2), padding = 'same'))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(2, (3, 3), activation = 'relu',  padding = 'same'))
model.add(layers.AveragePooling2D((2, 2), padding = 'same'))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(4, (3, 3), activation = 'relu',  padding = 'same'))
model.add(layers.AveragePooling2D((2, 2), padding = 'same'))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(8, (3, 3), activation = 'relu',  padding = 'same'))
model.add(layers.AveragePooling2D((2, 2), padding = 'same'))
model.add(layers.BatchNormalization())

model.add(layers.Flatten())
model.add(layers.Dense(slice_size*slice_size, activation = 'sigmoid'))
model.add(layers.Reshape((slice_size, slice_size, 1)))

# %% codecell
# train the model
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = False)
model.compile(optimizer = 'adam', loss = loss_fn, metrics = ['accuracy'])
history = model.fit(ds_train, epochs = num_epochs, verbose = 2,
    batch_size = batch_size, validation_data = ds_test)
model.save("models/predictor_" + run + "_v" + str(version) + "_m"
    + str(model_num) + ".pb")

# %% codecell
# plot accuracy progression
plt.plot(history.history['accuracy'], label = 'Train Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')
plt.show()

# %% codecell
# check a prediction
im = random.randint(0, len(gds_paths))
gds, sem = read_image(gds_paths[im], sem_paths[im])
prediction = np.squeeze(model(gds[None, ...]).numpy())

# binarize the prediction
prediction_bin = cv2.GaussianBlur(prediction, (5, 5), 0)
prediction_bin[prediction_bin >= 0.5] = 1
prediction_bin[prediction_bin < 0.5] = 0

# show one example
plt.figure(figsize = (8, 9))
plt.subplot(2, 2, 1)
plt.title("Nominal")
plt.imshow(gds)
plt.subplot(2, 2, 2)
plt.title("Ideal")
plt.imshow(sem)
plt.subplot(2, 2, 3)
plt.title("Raw Prediction")
plt.imshow(prediction, vmin = 0, vmax = 1)
plt.subplot(2, 2, 4)
plt.title("Binarized Prediction")
plt.imshow(prediction_bin)
plt.show()
