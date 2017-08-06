import sys
from helpers import *
from models import NaiveModel

# Settings
epochs = 100
batch_size = 16
img_width, img_height = 150, 150

# Get class weights
female_train, female_test, male_train, male_test = get_data()
class_weight = {
    0: 1.,
    1: float(len(female_train))/float(len(male_train))
}

# Train
model = NaiveModel(tmp_dir, 'naive', img_width, img_height)
if len(sys.argv) == 2:
    model.load(sys.argv[1])
model.train(train_dir, test_dir, epochs, batch_size, class_weight)
model.save(os.path.join(results_dir, 'naive.h5'))
