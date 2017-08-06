import sys
from helpers import *
from models import NaiveCNN, VGG16

# Inputs
name = sys.argv[1]
checkpoint = sys.argv[2]

# Predict
if name == 'naive_cnn':
    model = NaiveCNN(tmp_dir, name)
elif name == 'vgg16':
    model = VGG16(tmp_dir, name)
else:
    raise NotImplementedError('Model '+name+' not implemented!')
model.load(checkpoint)
predictions = model.predict(test_dir)

# Evaluate
evaluate(name, predictions)
