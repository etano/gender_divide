"""Create directories and copy files into respective directories"""

import sys
from helpers import *

def make_directories(base_dir, dirs):
    """Make directories in base_dir given list of directory paths

    Args:
        base_dir (str): Path to base data directory
        dirs (list(str)): Directories to make
    """
    for dir in dirs:
        if not os.path.exists(os.path.join(base_dir, dir)):
            os.makedirs(os.path.join(base_dir, dir))

def copy_files(files, dir):
    """Copy files into directory that may or may not exist

    Args:
        files (list(str)): List of paths to files
        dir (str): Path to target directory
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    for file in files:
        shutil.copy2(file, dir)

# Copy files into directory structure
def copy_files_into_directories(data_dir):
    """Copy image files into train and test directories that may or may not exist

    Args:
        data_dir (str): Path to base data directory
    """
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
