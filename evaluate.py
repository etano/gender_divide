import sys
from helpers import *
from models import NaiveCNN, VGG16

# Inputs
name = sys.argv[1]
model_type = sys.argv[2]
data_dir = sys.argv[3]
checkpoint = sys.argv[4]

# Predict
if model_type == 'naive_cnn':
    model = NaiveCNN(tmp_dir, name)
elif model_type == 'vgg16':
    model = VGG16(tmp_dir, name)
else:
    raise NotImplementedError('Model '+model_type+' not implemented!')
model.load(checkpoint)
predictions = model.predict(test_dir)

# Evaluate
evaluate(name, predictions)
