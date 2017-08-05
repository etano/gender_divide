"""Global things"""

import os, json

# Directories
img_dir = './data/img'
meta_dir = './data/meta'
train_dir = './data/train'
test_dir = './data/test'
weights_dir = './models'

# Get metadata (removes duplicates)
def get_gender_metadata(dir, file):
    with open(file) as x:
        data = json.load(x)
        return [os.path.join(dir, f) for f in set(data['female'])], [os.path.join(dir, f) for f in set(data['male'])]

# Get data
def get_data():
    female_train, male_train = get_gender_metadata(img_dir, os.path.join(meta_dir, 'train.json'))
    female_test, male_test = get_gender_metadata(img_dir, os.path.join(meta_dir, 'test.json'))
    return female_train, female_test, male_train, male_test
