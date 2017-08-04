import os, json, shutil, cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# Global
img_dir = './data/img'
meta_dir = './data/meta'
train_dir = './data/train'
test_dir = './data/test'

# Get metadata (removes duplicates)
def get_gender_metadata(file):
    with open(file) as x:
        data = json.load(x)
        return [os.path.join(img_dir, f) for f in set(data['female'])], [os.path.join(img_dir, f) for f in set(data['male'])]
female_train, male_train = get_gender_metadata(os.path.join(meta_dir, 'train.json'))
female_test, male_test = get_gender_metadata(os.path.join(meta_dir, 'test.json'))
train_all = female_train + male_train
test_all = female_test + male_test

# Copy files into respective directories
def copy_files(files, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for file in files:
        shutil.copy2(file, directory)
copy_files(female_train, os.path.join(train_dir, 'female'))
copy_files(female_test, os.path.join(test_dir, 'female'))
copy_files(male_train, os.path.join(train_dir, 'male'))
copy_files(male_test, os.path.join(test_dir, 'male'))

# dimensions of our images.
img_width, img_height = 150, 150
nb_train_samples = len(train_all)
nb_test_samples = len(test_all)
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# class weights to balance training
class_weight = {
    train_generator.class_indices['female'] : 1.,
    train_generator.class_indices['male']: float(len(female_train))/float(len(male_train))
}

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=nb_test_samples // batch_size,
    class_weight=class_weight)

model.save_weights('naive.h5')
