from model import *
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

class TransferModel(Model):
    """Transfer learning model (train on bottleneck features)

    Attributes:
        model (Keras model): Keras model
        bottleneck_model (Keras model): Keras model
        weights_dir (str): Weights directory
        name (str): Name of model
        img_width (int): Image width
        img_height (int): Image height
    """

    def __init__(self, weights_dir, name='naive', img_width=150, img_height=150):
        """Create model

        Args:
            weights_dir (str): Weights directory
            name (str): Name of model
            img_width (int): Image width
            img_height (int): Image height
        """
        super(TransferModel, self).__init__(weights_dir, name, img_width, img_height)

        self.bottleneck_model = applications.VGG16(include_top=False, weights='imagenet')
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(4, 4, self.bottleneck_model.output_shape[-1])))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, train_dir, test_dir, epochs=50, batch_size=16, class_weight=None):
        """Trains the model

        Args:
            train_dir (str): Directory with training images (organized into classes)
            test_dir (str): Directory with testing images (organized into classes)
            epochs (int): Number of epochs
            batch_size (int): Batch size
            class_weights (dict): Dictionary with class weights
        """

        train_datagen = ImageDataGenerator(
            rescale = 1./255,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True
        )
        features_file, labels_file = self.save_bottleneck_features(train_dir, 'train', train_datagen)
        train_data = np.load(open(features_file))
        train_labels = np.load(open(labels_file))

        features_file, labels_file = self.save_bottleneck_features(test_dir, 'test')
        test_data = np.load(open(features_file))
        test_labels = np.load(open(labels_file))

        self.model.fit(
            train_data,
            train_labels,
            epochs = epochs,
            batch_size = batch_size,
            validation_data = (test_data, test_labels),
            class_weight = class_weight
        )

    def predict(self, dir, batch_size=16):
        """Predicts using the model

        Args:
            dir (str): Directory with images to predict
            batch_size (int): Batch size

        Returns:
            list(list(str, int, np.array)): List of (filename, class, prediction) lists
        """
        filenames_file, features_file, labels_file = self.save_bottleneck_features(dir, 'predict')
        filenames = np.load(open(filenames_file))
        features = np.load(open(features_file))
        labels = np.load(open(labels_file))

        predictions = self.model.predict(
            features,
            batch_size = batch_size
        )

        return zip(filenames, labels, predictions)

    def save_bottleneck_features(self, dir, suffix, datagen=ImageDataGenerator(rescale=1./255), overwrite=False):
        """Saves features coming from the bottleneck model

        Args:
            dir (str): Directory with images (organized into classes)
            suffix (str): Suffix for saved feature files
            datagen (Keras ImageDataGenerator): Keras image data generator
            overwrite (bool): Whether or not to overwrite existing files

        Returns:
            str: Path of filenames file
            str: Path of features file
            str: Path of labels file
        """
        filenames_file = os.path.join(self.weights_dir, 'bottleneck_filenames_'+suffix+'.npy')
        features_file = os.path.join(self.weights_dir, 'bottleneck_features_'+suffix+'.npy')
        labels_file = os.path.join(self.weights_dir, 'bottleneck_labels_'+suffix+'.npy')

        if overwrite or (not os.path.exists(filenames_file)) or (not os.path.exists(features_file)) or (not os.path.exists(labels_file)):
            generator = datagen.flow_from_directory(
                dir,
                target_size = (self.img_width, self.img_height),
                batch_size = 1,
                class_mode = None,
                shuffle = False
            )
            filenames = generator.filenames
            features = self.bottleneck_model.predict_generator(generator, generator.samples)
            labels = generator.classes
            np.save(open(filenames_file, 'w'), filenames)
            np.save(open(features_file, 'w'), features)
            np.save(open(labels_file, 'w'), labels)

        return filenames_file, features_file, labels_file
