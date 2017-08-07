import sys
from helpers import *
from models import NaiveCNN, VGG16

# Inputs
name = sys.argv[1]
model_type = sys.argv[2]
data_dir = sys.argv[3]
checkpoint = sys.argv[4] if len(sys.argv) > 4 else None

# Get class weights
female_train, female_test, male_train, male_test = get_data(data_dir)
class_weight = {
    0: float(len(male_train))/float(len(female_train)),
    1: 1.
}
print class_weight

# Train
if model_type == 'naive_cnn':
    model = NaiveCNN(tmp_dir, name)
elif model_type == 'vgg16':
    model = VGG16(tmp_dir, name)
else:
    raise NotImplementedError('Model '+model_type+' not implemented!')
if checkpoint != None:
    model.load(checkpoint)
img_dir, meta_dir, train_dir, test_dir = get_directories(data_dir)
model.train(train_dir, test_dir, class_weight=class_weight)
model.save(os.path.join(results_dir, name+'.h5'))
