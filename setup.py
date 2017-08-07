"""Create directories and copy files into respective directories"""

import shutil
from helpers import *

# Make global directories
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

# Make data directories
def make_directories(data_dir):
    img_dir, meta_dir, train_dir, test_dir = get_directories(data_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
make_directories('./data')
make_directories('./data/amazon')

# Copy files into directory structure
def copy_gender_images(data_dir, copy_test):
    print 'Copying images from', data_dir
    img_dir, meta_dir, train_dir, test_dir = get_directories(data_dir)
    female_train, female_test, male_train, male_test = get_data(data_dir)
    copy_files(female_train, os.path.join(train_dir, 'female'))
    copy_files(male_train, os.path.join(train_dir, 'male'))
    if copy_test:
        copy_files(female_test, os.path.join(test_dir, 'female'))
        copy_files(male_test, os.path.join(test_dir, 'male'))
copy_gender_images('./data', True)
copy_gender_images('./data/amazon', False)
