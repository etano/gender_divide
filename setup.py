"""Create directories and copy files into respective directories"""

import sys
from helpers import *

# Copy files into directory structure
def copy_files_into_directories(data_dir):
    print 'Copying files into directories in', data_dir
    img_dir, meta_dir, train_dir, test_dir = get_directories(data_dir)
    female_train, female_test, male_train, male_test = get_data(data_dir)
    copy_files(female_train, os.path.join(train_dir, 'female'))
    copy_files(male_train, os.path.join(train_dir, 'male'))
    copy_files(female_test, os.path.join(test_dir, 'female'))
    copy_files(male_test, os.path.join(test_dir, 'male'))

def usage():
    print """
USAGE: python parse.py PATH_TO_DATA_DIR
"""
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
    data_dir = sys.argv[1]
    make_directories(data_dir, ['train', 'test'])
    copy_files_into_directories(data_dir)
