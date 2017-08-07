from model import *
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras import applications
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

class VGG16(Model):
    """Transfer learning model starting from VGG16 (train on bottleneck features)

    Attributes:
        model (Keras model): Keras model
        bottleneck_model (Keras model): Keras model
        weights_dir (str): Weights directory
        name (str): Name of model
        img_width (int): Image width
        img_height (int): Image height
        suffix (str): Suffix for all saved files
    """

    def __init__(self, weights_dir, name='vgg16', img_width=224, img_height=224):
        """Create model

        Args:
            weights_dir (str): Weights directory
            name (str): Name of model
            img_width (int): Image width
            img_height (int): Image height
        """
        super(VGG16, self).__init__(weights_dir, name, img_width, img_height)

        self.bottleneck_model = applications.VGG16(include_top=False, weights='imagenet')
        dummy_img = np.zeros((1,img_width, img_height, 3))
        features = self.bottleneck_model.predict(dummy_img, batch_size=1)

        self.model = Sequential()
        self.model.add(Flatten(input_shape=features.shape[1:]))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(
            optimizer = 'adam',
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
        )

        self.suffix = '_'+self.name+'_'+str(self.img_width)+'_'+str(self.img_height)

    def train(self, train_dir, test_dir, epochs=50, batch_size=16, class_weight=None):
        """Trains the model

        Args:
            train_dir (str): Directory with training images (organized into classes)
            test_dir (str): Directory with testing images (organized into classes)
            epochs (int): Number of epochs
            batch_size (int): Batch size
            class_weights (dict): Dictionary with class weights
        """

        train_datagen = ImageDataGenerator(rescale = 1./255)
        filenames_file, features_file, labels_file = self.save_bottleneck_features(train_dir, 'train'+self.suffix, train_datagen)
        train_data = np.load(open(features_file))
        train_labels = np.load(open(labels_file))

        filenames_file, features_file, labels_file = self.save_bottleneck_features(test_dir, 'test'+self.suffix)
        test_data = np.load(open(features_file))
        test_labels = np.load(open(labels_file))

        # Checkpointing
        checkpoint = ModelCheckpoint(
            os.path.join(self.weights_dir, self.name+'_{epoch:02d}_{val_acc:.2f}.h5'),
            monitor = 'val_acc',
            verbose = 1,
            save_best_only = False,
            mode = 'max'
        )

        # Logging
        csv_logger = CSVLogger(os.path.join(self.weights_dir, self.name+'_log.csv'), append=True, separator=',')

        # Fit
        callbacks_list = [checkpoint, csv_logger]
        self.model.fit(
            train_data,
            train_labels,
            epochs = epochs,
            batch_size = batch_size,
            validation_data = (test_data, test_labels),
            class_weight = class_weight,
            callbacks = callbacks_list,
            shuffle = True
        )

    def predict(self, dir, batch_size=16):
        """Predicts using the model

        Args:
            dir (str): Directory with images to predict
            batch_size (int): Batch size

        Returns:
            list(list(str, int, np.array)): List of (filename, class, prediction) lists
        """
        filenames_file, features_file, labels_file = self.save_bottleneck_features(dir, 'predict'+self.suffix)
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
