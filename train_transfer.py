import sys
from helpers import *
from models import TransferModel

# Settings
epochs = 100
batch_size = 16
img_width, img_height = 224, 224

# Get class weights
female_train, female_test, male_train, male_test = get_data()
class_weight = {
    0: float(len(male_train))/float(len(female_train)),
    1: 1.
}

# Train
model = TransferModel(tmp_dir, 'transfer', img_width, img_height)
model.train(train_dir, test_dir, epochs, batch_size, class_weight)
model.save(os.path.join(results_dir, 'transfer.h5'))
