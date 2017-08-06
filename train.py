import sys
from helpers import *
from models import NaiveCNN, VGG16

# Inputs
name = sys.argv[1]
checkpoint = sys.argv[2] if len(sys.argv) > 2 else None

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
if name == 'naive_cnn':
    model = NaiveCNN(tmp_dir, name, img_width, img_height)
elif name == 'vgg16':
    model = VGG16(tmp_dir, name, img_width, img_height)
else:
    raise NotImplementedError('Model '+name+' not implemented!')
if checkpoint != None:
    model.load(checkpoint)
model.train(train_dir, test_dir, epochs, batch_size, class_weight)
model.save(os.path.join(results_dir, name+'.h5'))
