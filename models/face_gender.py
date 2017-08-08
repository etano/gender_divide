import os, cv2
from imutils import paths
from keras.models import load_model
from keras.preprocessing import image
from model import *

class FaceGender(Model):
    """Get gender from face after face detection

    Attributes:
        model (Keras model): Keras model
        weights_dir (str): Weights directory
        name (str): Name of model
        img_width (int): Image width
        img_height (int): Image height
    """

    def __init__(self, cascade_classifier_weights, gender_classifier_weights, weights_dir, name='face_gender', img_width=150, img_height=150):
        """Create model

        Args:
            weights_dir (str): Weights directory
            name (str): Name of model
            img_width (int): Image width
            img_height (int): Image height
        """
        super(FaceGender, self).__init__(weights_dir, name, img_width, img_height)

        # Load classifers
        self.cc = cv2.CascadeClassifier(cascade_classifier_weights)
        self.gender_classifier = load_model(gender_classifier_weights)

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
        gender_target_size = self.gender_classifier.input_shape[1:3]
        gender_offsets = (10, 10)
        filenames, labels, predictions = [], [], []
        for image_path in paths.list_images(dir):
            split_path = image_path.split('/')
            filename = split_path[-1]
            gender = split_path[-2]
            filenames.append(gender+'/'+filename)
            label = 0 if (gender == 'female') else 1
            labels.append(label)

            rgb_image = self.load_image(image_path)
            gray_image = self.load_image(image_path, grayscale=True)
            gray_image = np.squeeze(gray_image)
            gray_image = gray_image.astype('uint8')

            coords = self.cc.detectMultiScale(gray_image, 1.3, 5)
            if len(coords) == 0:
                predictions.append(None)
                continue
            coords = self.get_largest_area(coords)
            x1, x2, y1, y2 = self.apply_offsets(coords, gender_offsets)
            rgb_face = rgb_image[y1:y2, x1:x2]

            try:
                rgb_face = cv2.resize(rgb_face, (gender_target_size))
            except:
                predictions.append(None)
                continue

            rgb_face = self.preprocess_input(rgb_face, False)
            rgb_face = np.expand_dims(rgb_face, 0)
            prediction = self.gender_classifier.predict(rgb_face)
            predictions.append(prediction[0,1])
        return zip(filenames, labels, predictions)

    def load_image(self, image_path, grayscale=False, target_size=None):
        pil_image = image.load_img(image_path, grayscale, target_size)
        return image.img_to_array(pil_image)

    def apply_offsets(self, coords, offsets):
        x, y, width, height = coords
        x_off, y_off = offsets
        return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

    def preprocess_input(self, x, v2=True):
        x = x.astype('float32')
        x = x / 255.0
        return x

    def get_largest_area(self, coords):
        largest_area = 0
        largest_coords = 0, 0, 0, 0
        for (x, y, width, height) in coords:
            area = width * height
            if area > largest_area:
                largest_area = area
                largest_coords = (x, y, width, height)
        return largest_coords
