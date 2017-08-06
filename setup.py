"""Create directories and copy files into respective directories"""

import shutil
from helpers import *

# Make directories
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
if not os.path.exists(amazon_dir):
    os.makedirs(amazon_dir)

# Copy files into directory structure
female_train, female_test, male_train, male_test = get_data()
def copy_files(files, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for file in files:
        shutil.copy2(file, directory)
copy_files(female_train, os.path.join(train_dir, 'female'))
copy_files(female_test, os.path.join(test_dir, 'female'))
copy_files(male_train, os.path.join(train_dir, 'male'))
copy_files(male_test, os.path.join(test_dir, 'male'))
