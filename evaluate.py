"""Evaluate a model"""

import sys
from helpers import *
from models import load_model

if len(sys.argv) < 4:
    print "USAGE: python evaluate.py NAME MODEL_TYPE DATA_DIR (CHECKPOINT_HDF5_FILE)"
    sys.exit(0)

# Inputs
name = sys.argv[1]
type = sys.argv[2]
data_dir = sys.argv[3]
checkpoint = sys.argv[4] if len(sys.argv) > 4 else None

# Load model
model = load_model(name, type, tmp_dir, checkpoint)

# Get predictions
img_dir, meta_dir, train_dir, test_dir = get_directories(data_dir)
predictions = model.predict(test_dir)

# Evaluate
evaluate(name, predictions, test_dir)
