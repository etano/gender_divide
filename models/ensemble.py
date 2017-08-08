import os, cv2
from imutils import paths
from model import *

class Ensemble(Model):
    """Ensemble some models (simple average)

    Attributes:
        model (Keras model): Keras model
        weights_dir (str): Weights directory
        name (str): Name of model
        img_width (int): Image width
        img_height (int): Image height
    """

    def __init__(self, models, weights_dir, name='ensemble', img_width=150, img_height=150):
        """Create model

        Args:
            weights_dir (str): Weights directory
            name (str): Name of model
            img_width (int): Image width
            img_height (int): Image height
        """
        super(Ensemble, self).__init__(weights_dir, name, img_width, img_height)

        # Load models
        self.models = models

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
        if len(self.models) == 0:
            print 'No models defined!'
            return None
        results = []
        for model in self.models:
            print 'Running', model.name, '...'
            predictions = model.predict(dir)
            predictions.sort(key=lambda x: x[0])
            results.append(predictions)
            print '...done.'
        filenames = [x[0] for x in results[0]]
        labels = [x[1] for x in results[0]]
        all_predictions = []
        for result in results:
            all_predictions.append([x[2] for x in result])
        t_all_predictions = map(list, zip(*all_predictions)) # transpose magic
        predictions = []
        weights = np.array([2., 1.])
        for i in range(len(filenames)):
            vals = np.array([np.squeeze(x) for x in t_all_predictions[i] if x != None])
            total = 0.
            for i in range(len(vals)):
                total += vals[i] * weights[i]
            predictions.append(total / np.sum(weights))
        return zip(filenames, labels, predictions)
