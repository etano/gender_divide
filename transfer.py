import os, json, shutil, cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# Global
img_dir = './data/img'
meta_dir = './data/meta'
train_dir = './data/train'
test_dir = './data/test'
top_model_weights_path = './weights/bottleneck_fc_model.h5'

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
epochs = 50
batch_size = 16
nb_train_samples = len(train_all)
nb_test_samples = len(test_all)

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    bottleneck_labels_train = np.array([0]*len(female_train) + [1]*len(male_train))
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
    np.save(open('bottleneck_labels_train.npy', 'w'), bottleneck_labels_train)

    generator = datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)
    bottleneck_features_test = model.predict_generator(generator, nb_test_samples)
    bottleneck_labels_test = np.array([0]*len(female_test) + [1]*len(male_test))
    np.save(open('bottleneck_features_test.npy', 'w'), bottleneck_features_test)
    np.save(open('bottleneck_labels_test.npy', 'w'), bottleneck_labels_test)

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.load(open('bottleneck_labels_train.npy'))

    test_data = np.load(open('bottleneck_features_test.npy'))
    test_labels = np.load(open('bottleneck_labels_test.npy'))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    print(train_data.shape[1:])
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # class weights to balance training
    class_weight = {
        0 : 1.,
        1: float(len(female_train))/float(len(male_train))
    }

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(test_data, test_labels),
              class_weight=class_weight)
    model.save_weights(top_model_weights_path)

save_bottlebeck_features()
train_top_model()
