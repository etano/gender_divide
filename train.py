import sys
from helpers import *
from models import NaiveCNN, VGG16

# Inputs
name = sys.argv[1]
data_dir = sys.argv[2]
checkpoint = sys.argv[3] if len(sys.argv) > 3 else None

# Get class weights
female_train, female_test, male_train, male_test = get_data(data_dir)
class_weight = {
    0: float(len(male_train))/float(len(female_train)),
    1: 1.
}
print class_weight

# Train
if name == 'naive_cnn':
    model = NaiveCNN(tmp_dir, name)
elif name == 'vgg16':
    model = VGG16(tmp_dir, name)
else:
    raise NotImplementedError('Model '+name+' not implemented!')
if checkpoint != None:
    model.load(checkpoint)
img_dir, meta_dir, train_dir, test_dir = get_directories(data_dir)
model.train(train_dir, test_dir, class_weight=class_weight)
model.save(os.path.join(results_dir, name+'.h5'))
