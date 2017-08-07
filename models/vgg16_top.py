from model import *
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D
from keras import applications
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras import optimizers
import keras

class VGG16Top(Model):
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
        super(VGG16Top, self).__init__(weights_dir, name, img_width, img_height)

        self.base_model = applications.VGG16(include_top=False, weights='imagenet')
        #dummy_img = np.zeros((1, img_width, img_height, 3))
        #features = self.bottleneck_model.predict(dummy_img, batch_size=1)

        #top_model = Sequential()
        #top_model.add(Flatten(input_shape=features.shape[1:]))
        #top_model.add(Dense(256, activation='relu'))
        #top_model.add(Dropout(0.5))
        #top_model.add(Dense(1, activation='sigmoid'))
        #top_model.load_weights(os.path.join(weights_dir, 'vgg16_top.h5'))

        # Top Model Block
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(1, activation='sigmoid')(x)

        # add your top layer block to your base model
        self.model = keras.models.Model(self.base_model.input, predictions)

        for layer in self.base_model.layers:
            layer.trainable = False

        self.model.compile(
            loss = 'binary_crossentropy',
            optimizer = optimizers.SGD(lr=1e-4, momentum=0.9),
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
