import sys
from helpers import *
from models import NaiveModel

# Settings
name = 'naive'
batch_size = 16
img_width, img_height = 150, 150

# Predict
model = NaiveModel(tmp_dir, name, img_width, img_height)
model.load(sys.argv[1])
predictions = model.predict(test_dir, batch_size)

# Evaluate
evaluate(name, predictions)
