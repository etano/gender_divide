"""Model base class"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class Model(object):
    """Model base class

    Attributes:
        model (Keras model): Keras model
        weights_dir (str): Weights directory
        name (str): Name of model
        img_width (int): Image width
        img_height (int): Image height
    """

    def __init__(self, weights_dir, name, img_width, img_height):
        """Create model

        Args:
            weights_dir (str): Weights directory
            name (str): Name of model
            img_width (int): Image width
            img_height (int): Image height
        """
        self.weights_dir = weights_dir
        self.name = name
        self.img_width = img_width
        self.img_height = img_height
        self.model = None

    def save(self, file=None):
        """Saves the model

        Args:
            file (str): File name
        """
        if file == None:
            file = os.path.join(self.weights_dir, self.name+'.h5')
        self.model.save_weights(file)

    def load(self, file=None):
        """Loads the model

        Args:
            file (str): File name
        """
        if file == None:
            file = os.path.join(self.weights_dir, self.name+'.h5')
        self.model.load_weights(file)

    def train(self, train_dir, test_dir, epochs=50, batch_size=16, class_weight=None):
        """Trains the model

        Args:
            train_dir (str): Directory with training images (organized into classes)
            test_dir (str): Directory with testing images (organized into classes)
            epochs (int): Number of epochs
            batch_size (int): Batch size
            class_weights (dict): Dictionary with class weights
        """
        raise NotImplementedError("Subclasses should implement this!")

    def predict(self, dir, batch_size=16):
        """Predicts using the model

        Args:
            dir (str): Directory with images to predict
            batch_size (int): Batch size

        Returns:
            list(list(str, int, np.array)): List of (filename, class, prediction) lists
        """
        raise NotImplementedError("Subclasses should implement this!")
