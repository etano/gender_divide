from model import *
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

class NaiveCNN(Model):
    """Simple convolutional model

    Attributes:
        model (Keras model): Keras model
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
        super(NaiveCNN, self).__init__(weights_dir, name, img_width, img_height)
        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(
            loss = 'binary_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy']
        )

    def train(self, train_dir, test_dir, epochs=50, batch_size=16, class_weight=None):
        """Trains the model

        Args:
            train_dir (str): Directory with training images (organized into classes)
            test_dir (str): Directory with testing images (organized into classes)
            epochs (int): Number of epochs
            batch_size (int): Batch size
            class_weights (dict): Dictionary with class weights
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale = 1./255,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True
        )
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size = (self.img_width, self.img_height),
            batch_size = batch_size,
            class_mode = 'binary'
        )

        # No data augmentation for testing
        test_datagen = ImageDataGenerator(rescale = 1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size = (self.img_width, self.img_height),
            batch_size = batch_size,
            class_mode = 'binary'
        )

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
        self.model.fit_generator(
            train_generator,
            steps_per_epoch = train_generator.samples // batch_size,
            epochs = epochs,
            validation_data = test_generator,
            validation_steps = test_generator.samples // batch_size,
            class_weight = class_weight,
            callbacks = callbacks_list
        )

    def predict(self, dir, batch_size=16):
        """Predicts using the model

        Args:
            dir (str): Directory with images to predict
            batch_size (int): Batch size

        Returns:
            list(list(str, int, np.array)): List of (filename, class, prediction) lists
        """
        datagen = ImageDataGenerator(rescale = 1./255)
        generator = datagen.flow_from_directory(
            dir,
            target_size = (self.img_width, self.img_height),
            batch_size = batch_size,
            class_mode = 'binary'
        )
        predictions = self.model.predict_generator(
            generator,
            generator.samples // batch_size
        )
        return zip(generator.filenames, generator.classes, predictions)
