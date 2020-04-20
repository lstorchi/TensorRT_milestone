import numpy as np
import tensorflow as tf
import pandas as pd
import h5py

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model

# Dataset reading
pixel_data = np.array(pd.read_hdf('pixel_only_data_test.h5', key='data'))
pixel_labels = np.array(pd.read_hdf('pixel_only_data_test.h5', key='labels'))

pixel_data = np.transpose(np.reshape(pixel_data, [-1,20,16,16]), (0,2,3,1))

pixel_train_data = pixel_data[:19000, ]
pixel_test_data = pixel_data[19000:, ]
pixel_train_labels = pixel_labels[:19000, ]
pixel_test_labels = pixel_labels[19000:, ]

train_ds = tf.data.Dataset.from_tensor_slices(
    (pixel_train_data, pixel_train_labels)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((pixel_test_data,  
                                              pixel_test_labels)).batch(32)


# Model Building
class pixel_only_model(Model):

    def __init__(self):
        super(pixel_only_model, self).__init__()
        self.conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1')
        self.conv2 = Conv2D(64, (3, 3), activation='relu', name='conv2')
        self.pool1 = MaxPooling2D(pool_size=(2, 2), padding='same', 
                                  name='pool1')
        self.conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                            name='conv3')
        self.conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                            name='conv4')
        self.pool2 = MaxPooling2D(pool_size=(2, 2), padding='same', 
                                  name='pool2')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu', name='dense1')
        self.d2 = Dense(64, activation='relu', name='dense2')
        self.d3 = Dense(2, activation='relu', name='output')

    def call(self, X):
        c1 = self.conv1(X)
        c2 = self.conv2(c1)
        p1 = self.pool1(c2)
        c3 = self.conv2(p1)
        c4 = self.conv2(c3)
        p2 = self.pool1(c4)
        f = self.flatten(p2)
        d1 = self.d1(f)
        d2 = self.d2(d1)
        out = self.d3(d2)
        return out


# Model, Optimizer and Losses defeinitions
model = pixel_only_model()

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.\
                 CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.\
                CategoricalAccuracy(name='test_accuracy')


# Training Function
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

# Test Function
@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).

    predictions = model(images, training=False)

    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


# Actual Trainig
EPOCHS = 5
for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, \
                Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))


tf.saved_model.save(model, 'pixel_only_model')
