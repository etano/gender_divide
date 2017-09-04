"""Make testing and training meta JSON file with paths to images based on directory of images"""

import sys, os, json, random
from helpers import *

if len(sys.argv) < 4:
    print "USAGE: python make_meta_json.py IMG_DIR TARGET_DIR TEST_FRACTION MAX_IMAGES"
    sys.exit(0)

# Inputs
img_dir = sys.argv[1]
target_dir = sys.argv[2]
test_fraction = float(sys.argv[3])
max_imgs = int(sys.argv[4])

# Get images
blank_meta = {'female': [], 'male': []}
test_meta = {
    'female': get_imgs('female', img_dir, blank_meta, test_fraction*max_imgs),
    'male': get_imgs('male', img_dir, blank_meta, test_fraction*max_imgs)
}
train_meta = {
    'female': get_imgs('female', img_dir, test_meta, max_imgs),
    'male': get_imgs('male', img_dir, test_meta, max_imgs)
}

# Write out JSON file
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
with open(os.path.join(target_dir, 'test.json'), 'w') as f:
    json.dump(test_meta, f, indent=4)
with open(os.path.join(target_dir, 'train.json'), 'w') as f:
    json.dump(train_meta, f, indent=4)
