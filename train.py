"""Train a model"""

import sys
from helpers import *
from models import load_model

if len(sys.argv) < 4:
    print "USAGE: python train.py NAME MODEL_TYPE DATA_DIR (CHECKPOINT_HDF5_FILE)"
    sys.exit(0)

# Inputs
name = sys.argv[1]
type = sys.argv[2]
data_dir = sys.argv[3]
checkpoint = sys.argv[4] if len(sys.argv) > 4 else None

# Load model
model = load_model(name, type, tmp_dir, checkpoint)

# Get class weights
female_train, female_test, male_train, male_test = get_data(data_dir)
class_weight = {
    0: float(len(male_train))/float(len(female_train)),
    1: 1.
}

# Train model
img_dir, meta_dir, train_dir, test_dir = get_directories(data_dir)
model.train(train_dir, test_dir, class_weight=class_weight)
model.save(os.path.join(results_dir, name+'.h5'))
