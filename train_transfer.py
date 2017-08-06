import sys
from helpers import *
from models import TransferModel

# Settings
epochs = 50
batch_size = 16
img_width, img_height = 150, 150

# Get class weights
female_train, female_test, male_train, male_test = get_data()
class_weight = {
    0: 1.,
    1: float(len(female_train))/float(len(male_train))
}

# Train
model = TransferModel(tmp_dir, 'transfer', img_width, img_height)
model.train(train_dir, test_dir, epochs, batch_size, class_weight)
model.save(os.path.join(results_dir, 'transfer.h5'))
