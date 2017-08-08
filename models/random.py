import os
from imutils import paths
from model import *

class Random(Model):
    """Random guessing

    Attributes:
        model (Keras model): Keras model
        weights_dir (str): Weights directory
        name (str): Name of model
        img_width (int): Image width
        img_height (int): Image height
    """

    def __init__(self, weights_dir, name='always_female', img_width=150, img_height=150):
        """Create model

        Args:
            weights_dir (str): Weights directory
            name (str): Name of model
            img_width (int): Image width
            img_height (int): Image height
        """
        super(Random, self).__init__(weights_dir, name, img_width, img_height)

    def load(self, file=None):
        """Loads the model

        Args:
            file (str): File name
        """
        pass

    def save(self, file=None):
        """Saves the model

        Args:
            file (str): File name
        """
        pass

    def train(self, train_dir, test_dir, epochs=50, batch_size=16, class_weight=None):
        """Trains the model

        Args:
            train_dir (str): Directory with training images (organized into labels)
            test_dir (str): Directory with testing images (organized into labels)
            epochs (int): Number of epochs
            batch_size (int): Batch size
            class_weights (dict): Dictionary with class weights
        """
        pass

    def predict(self, dir, batch_size=16):
        """Predicts using the model

        Args:
            dir (str): Directory with images to predict
            batch_size (int): Batch size

        Returns:
            list(list(str, int, np.array)): List of (filename, class, prediction) lists
        """
        filenames, labels = [], []
        for image_path in paths.list_images(dir):
            split_path = image_path.split('/')
            filename = split_path[-1]
            gender = split_path[-2]
            filenames.append(gender+'/'+filename)
            label = 0 if (gender == 'female') else 1
            labels.append(label)
        predictions = np.random.uniform(low=0.0, high=1.0, size=(len(filenames),))
        return zip(filenames, np.array(labels), predictions)
