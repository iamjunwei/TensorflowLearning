# https://developers.google.com/machine-learning/practica/image-classification/
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import random
# import os
# import matplotlib.pyplot as plt

train_dir = "./cats_and_dogs_filtered/train"
validation_dir = "./cats_and_dogs_filtered/validation"
train_cats_dir = "./cats_and_dogs_filtered/train/cats"
train_dogs_dir = "./cats_and_dogs_filtered/train/dogs"
validation_cats_dir = "./cats_and_dogs_filtered/validation/cats"
validation_dogs_dir = "./cats_and_dogs_filtered/validation/dogs"


# Step1 build a model
input_layer = tf.keras.layers.Input(shape=(150, 150, 3))
x = tf.keras.layers.Conv2D(16, 3, activation="relu")(input_layer)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(input_layer, output_layer)
model.summary()
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss=tf.keras.metrics.binary_crossentropy,
              metrics=["acc"])

train_datagen = ImageDataGenerator(rescale=1/255.0)
validation_datagen = ImageDataGenerator(rescale=1/255.0)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode="binary")
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(150, 150),
                                                              batch_size=20,
                                                              class_mode="binary")
history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=15,
                              validation_data=validation_generator,
                              validation_steps=50,
                              verbose=2)
acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
# epochs = range(len(acc))
# plt.figure()
# plt.plot(epochs, acc)
# plt.plot(epochs, val_acc)
# plt.title('Training and validation accuracy step1')
# plt.show()
# plt.figure()
# plt.plot(epochs, loss)
# plt.plot(epochs, val_loss)
# plt.title('Training and validation loss step1')
# plt.show()
print("Step1 training accuracy: ", acc[-1])
print("Step1 validation accuracy: ", val_acc[-1])
print("Step1 training loss: ", loss[-1])
print("Step1 validation loss: ", val_loss[-1])

# Step2 prevent overfitting
train_datagen = ImageDataGenerator(rescale=1/255.0,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode="binary")
history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=15,
                              validation_data=validation_generator,
                              validation_steps=50,
                              verbose=2)
acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
# epochs = range(len(acc))
# plt.figure()
# plt.plot(epochs, acc)
# plt.plot(epochs, val_acc)
# plt.title('Training and validation accuracy step2 data augmentation')
# plt.show()
# plt.figure()
# plt.plot(epochs, loss)
# plt.plot(epochs, val_loss)
# plt.title('Training and validation loss step2 data augmentation')
# plt.show()
print("Step2 data augmentation training accuracy: ", acc[-1])
print("Step2 data augmentation validation accuracy: ", val_acc[-1])
print("Step2 data augmentation training loss: ", loss[-1])
print("Step2 data augmentation validation loss: ", val_loss[-1])

input_layer = tf.keras.layers.Input(shape=(150, 150, 3))
x = tf.keras.layers.Conv2D(16, 3, activation="relu")(input_layer)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(input_layer, output_layer)
model.summary()
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss=tf.keras.metrics.binary_crossentropy,
              metrics=["acc"])
train_datagen = ImageDataGenerator(rescale=1/255.0)
validation_datagen = ImageDataGenerator(rescale=1/255.0)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode="binary")
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(150, 150),
                                                              batch_size=20,
                                                              class_mode="binary")
history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=15,
                              validation_data=validation_generator,
                              validation_steps=50,
                              verbose=2)
acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
# epochs = range(len(acc))
# plt.figure()
# plt.plot(epochs, acc)
# plt.plot(epochs, val_acc)
# plt.title('Training and validation accuracy step2 dropout')
# plt.show()
# plt.figure()
# plt.plot(epochs, loss)
# plt.plot(epochs, val_loss)
# plt.title('Training and validation loss step2 dropout')
# plt.show()
print("Step2 dropout training accuracy: ", acc[-1])
print("Step2 dropout validation accuracy: ", val_acc[-1])
print("Step2 dropout training loss: ", loss[-1])
print("Step2 dropout validation loss: ", val_loss[-1])

# Step3 Leveraging pretrained model
local_weight_file = "./inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
pre_trained_model = tf.keras.applications.InceptionV3(input_shape=(150, 150, 3),
                                                      include_top=False,
                                                      weights=None)
pre_trained_model.load_weights(local_weight_file)
for layer in pre_trained_model.layers:
    layer.trainable = False
# bottleneck layer is 3x3 feature map, we use the "mixed7" layer, which is 7x7 feature map
last_layer = pre_trained_model.get_layer("mixed7")
last_output = last_layer.output
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(pre_trained_model.input, x)
model.compile(optimizer=tf.train.RMSPropOptimizer(0.0001),
              loss=tf.keras.metrics.binary_crossentropy,
              metrics=["acc"])
train_datagen = ImageDataGenerator(rescale=1/255.0,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1/255.0)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode="binary")
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(150, 150),
                                                              batch_size=20,
                                                              class_mode="binary")
history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=2,
                              validation_data=validation_generator,
                              validation_steps=50,
                              verbose=2)
acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
# epochs = range(len(acc))
# plt.figure()
# plt.plot(epochs, acc)
# plt.plot(epochs, val_acc)
# plt.title('Training and validation accuracy step3')
# plt.show()
# plt.figure()
# plt.plot(epochs, loss)
# plt.plot(epochs, val_loss)
# plt.title('Training and validation loss step3')
# plt.show()
print("Step3 training accuracy: ", acc[-1])
print("Step3 validation accuracy: ", val_acc[-1])
print("Step3 training loss: ", loss[-1])
print("Step3 validation loss: ", val_loss[-1])

# Step4 Further more, Fine-Tuning
unfreeze = False
for layer in pre_trained_model.layers:
    if unfreeze:
        layer.trainable = True
    if layer.name == "mixed6":
        unfreeze = True
    print("{} trainable: {}".format(layer.name, layer.trainable))
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.SGD(lr=0.00001, momentum=0.9),
              metrics=['acc'])
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)
acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
# epochs = range(len(acc))
# plt.figure()
# plt.plot(epochs, acc)
# plt.plot(epochs, val_acc)
# plt.title('Training and validation accuracy step4')
# plt.show()
# plt.figure()
# plt.plot(epochs, loss)
# plt.plot(epochs, val_loss)
# plt.title('Training and validation loss step4')
# plt.show()
print("Step4 training accuracy: ", acc[-1])
print("Step4 validation accuracy: ", val_acc[-1])
print("Step4 training loss: ", loss[-1])
print("Step4 validation loss: ", val_loss[-1])
