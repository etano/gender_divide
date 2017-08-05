import sys
from helpers import *
from naive import NaiveModel

# Settings
epochs = 50
batch_size = 16
img_width, img_height = 150, 150

# Get class weights
female_train, female_test, male_train, male_test = get_data()
class_weight = {
    0 : 1.,
    1: float(len(female_train))/float(len(male_train))
}

# Train
model = NaiveModel(weights_dir, 'naive', img_width, img_height)
model.train(train_dir, test_dir, epochs, batch_size, class_weight)
model.save()
