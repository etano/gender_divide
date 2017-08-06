import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

class NaiveModel(object):
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
        self.weights_dir = weights_dir
        self.name = name
        self.img_width = img_width
        self.img_height = img_height
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

    def save(self, file=None):
        """Saves the model

        Args:
            file (str): File name
        """
        if file == None:
            file = os.path.join(self.weights_dir, self.name+'naive.h5')
        self.model.save_weights(file)

    def load(self, file=None):
        """Loads the model

        Args:
            file (str): File name
        """
        if file == None:
            file = os.path.join(self.weights_dir, self.name+'naive.h5')
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
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale = 1./255,
            shear_range = 0.2,
            zoom_range = 0.2,
            channel_shift_range = 0.2,
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
            os.path.join(self.weights_dir, self.name+'-{epoch:02d}-{val_acc:.2f}.h5'),
            monitor = 'val_acc',
            verbose = 1,
            save_best_only = False,
            mode = 'max'
        )
        callbacks_list = [checkpoint]

        # Fit
        self.model.fit_generator(
            train_generator,
            steps_per_epoch = train_generator.samples // batch_size,
            epochs = epochs,
            validation_data = test_generator,
            validation_steps = test_generator.samples // batch_size,
            class_weight = class_weight,
            callbacks = callbacks_list
        )

    def evaluate(self, dir, batch_size=16):
        """Evaluated using the model

        Args:
            dir (str): Directory with images to evaluate (organized into classes)
            batch_size (int): Batch size

        Returns:
            np.array: Numpy array with losses
        """
        datagen = ImageDataGenerator(rescale = 1./255)
        generator = datagen.flow_from_directory(
            dir,
            target_size = (self.img_width, self.img_height),
            batch_size = batch_size,
            class_mode = 'binary'
        )
        return self.model.evaluate_generator(
            generator,
            generator.samples // batch_size
        )

    def predict(self, dir, batch_size=16):
        """Predicts using the model

        Args:
            dir (str): Directory with images to predict
            batch_size (int): Batch size

        Returns:
            np.array: Numpy array with predictions
        """
        datagen = ImageDataGenerator(rescale = 1./255)
        generator = datagen.flow_from_directory(
            dir,
            target_size = (self.img_width, self.img_height),
            batch_size = batch_size,
            class_mode = 'binary'
        )
        return self.model.predict_generator(
            generator,
            generator.samples // batch_size
        )
