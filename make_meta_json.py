"""Make meta JSON file with paths to images based on directory of images, while avoiding test images"""

import sys, os, json, random
from helpers import *

if len(sys.argv) < 4:
    print "USAGE: python make_meta_json.py IMG_DIR TARGET_DIR PATH_TO_TEST_META_JSON (MAX_IMAGES)"
    sys.exit(0)

# Inputs
img_dir = sys.argv[1]
target_dir = sys.argv[2]
test_meta = json.load(open(sys.argv[3], 'r'))
max_imgs = int(sys.argv[4]) if len(sys.argv) > 4 else None

# Get images
train_meta = {
    'female': get_imgs('female', img_dir, max_imgs),
    'male': get_imgs('male', img_dir, max_imgs)
}

# Write out JSON file
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
with open(os.path.join(target_dir, 'train.json'), 'w') as f:
    json.dump(train_meta, f, indent=4)
