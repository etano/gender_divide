"""Project helpers"""

import os, json, shutil, random
import numpy as np

# GLOBAL VARIABLES
results_dir = './results'
tmp_dir = './tmp'

def get_directories(data_dir):
    """Get directory paths

    Args:
        data_dir (str): Path to base data directory

    Returns:
        str: Path to directory with all images
        str: Path to directory with meta json files
        str: Path to directory with training images
        str: Path to directory with testing images
    """
    img_dir = os.path.join(data_dir, 'img')
    meta_dir = os.path.join(data_dir, 'meta')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    return img_dir, meta_dir, train_dir, test_dir

def get_gender_metadata(dir, file):
    """Get gender metadata

    Args:
        dir (str): Path to base data directory
        file (str): Path to json file with metadata

    Returns:
        list(str): List of paths to female images
        list(str): List of paths to male images
    """
    with open(file) as x:
        data = json.load(x)
        return [os.path.join(dir, f) for f in set(data['female'])], [os.path.join(dir, f) for f in set(data['male'])]

def get_data(data_dir):
    """Get paths to images for gender classication

    Args:
        data_dir (str): Path to base data directory

    Returns:
        list(str): List of paths to female training images
        list(str): List of paths to female testing images
        list(str): List of paths to male training images
        list(str): List of paths to male testing images
    """
    img_dir, meta_dir, train_dir, test_dir = get_directories(data_dir)
    female_train, male_train = get_gender_metadata(img_dir, os.path.join(meta_dir, 'train.json'))
    female_test, male_test = get_gender_metadata(img_dir, os.path.join(meta_dir, 'test.json'))
    return female_train, female_test, male_train, male_test

def make_directories(base_dir, dirs):
    """Make directories in base_dir given list of directory paths

    Args:
        base_dir (str): Path to base data directory
        dirs (list(str)): Directories to make
    """
    for dir in dirs:
        if not os.path.exists(os.path.join(base_dir, dir)):
            os.makedirs(os.path.join(base_dir, dir))

def evaluate(name, predictions, test_dir):
    """Parse a set of predictions coming from test_dir,
       copy images into new directories accordingly,
       print confusion matrix,
       and compute statistics

    Args:
        name (str): Arbitrary model name
        predictions (list(list(str, int, float))): List of lists containing filenames, labels, and predictions
        test_dir (str): Path to directory with images that were used for prediction task
    """
    base_dir = os.path.join(tmp_dir, name)
    make_directories(base_dir, ['female/female', 'male/male', 'male/female', 'female/male', 'female/none', 'male/none'])
    confusion_matrix = np.zeros((2,2))
    total_male, total_female = 0, 0
    for [file, label, p] in predictions:
        if label == 1:
            total_male += 1
            if p == None:
                path = 'male/none'
            elif p > 0.5:
                confusion_matrix[1,1] += 1
                path = 'male/male'
            else:
                confusion_matrix[1,0] += 1
                path = 'male/female'
        else:
            total_female += 1
            if p == None:
                path = 'female/none'
            elif p <= 0.5:
                confusion_matrix[0,0] += 1
                path = 'female/female'
            else:
                confusion_matrix[0,1] += 1
                path = 'female/male'
        shutil.copy2(os.path.join(test_dir, file), os.path.join(base_dir, path))
    print(confusion_matrix)
    print('accuracy:', np.trace(confusion_matrix)/np.sum(confusion_matrix))
    print('female precision:', confusion_matrix[0,0]/np.sum(confusion_matrix[0,:]))
    print('male precision:', confusion_matrix[1,1]/np.sum(confusion_matrix[1,:]))
    print('female recall:', confusion_matrix[0,0]/total_female)
    print('male recall:', confusion_matrix[1,1]/total_male)

def get_imgs(gender, dir, test_meta, max_imgs=None):
    """Get list of images excluding ones in test_meta

    Args:
        gender (str): Gender of images
        dir (str): Path to base directory of images
        test_meta (json): JSON object with images to avoid
        max_imgs (int): (optional) Maximum number of images to return

    Returns:
        list(str): Relative paths to images (starting from dir)
    """
    imgs = []
    for img in os.listdir(os.path.join(dir, gender)):
        if not (gender+'/'+img in test_meta[gender]):
            imgs.append(gender+'/'+img)
    if max_imgs == None:
        return imgs
    else:
        random.shuffle(imgs)
        return imgs[:max_imgs]
