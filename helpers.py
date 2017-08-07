"""Global things"""

import os, json, shutil
import numpy as np

# Directories
results_dir = './results'
tmp_dir = './tmp'
def get_directories(data_dir):
    img_dir = os.path.join(data_dir, 'img')
    meta_dir = os.path.join(data_dir, 'meta')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    return img_dir, meta_dir, train_dir, test_dir

# Get metadata (removes duplicates)
def get_gender_metadata(dir, file):
    with open(file) as x:
        data = json.load(x)
        return [os.path.join(dir, f) for f in set(data['female'])], [os.path.join(dir, f) for f in set(data['male'])]

# Get data
def get_data(data_dir):
    img_dir, meta_dir, train_dir, test_dir = get_directories(data_dir)
    female_train, male_train = get_gender_metadata(img_dir, os.path.join(meta_dir, 'train.json'))
    female_test, male_test = get_gender_metadata(img_dir, os.path.join(meta_dir, 'test.json'))
    return female_train, female_test, male_train, male_test

# Make directories from list of directories
def makedirs(base_dir, dirs):
    for dir in dirs:
        if not os.path.exists(os.path.join(base_dir, dir)):
            os.makedirs(os.path.join(base_dir, dir))

# Copy files
def copy_files(files, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for file in files:
        shutil.copy2(file, directory)

# Evaluate predictions
def evaluate(name, predictions):
    base_dir = os.path.join(tmp_dir, name)
    makedirs(base_dir, ['female/female', 'male/male', 'male/female', 'female/male'])
    confusion_matrix = np.zeros((2,2))
    for [file, c, p] in predictions:
        if c == 1:
            if p >= 0.5:
                confusion_matrix[1,1] += 1
                path = 'male/male'
            else:
                confusion_matrix[1,0] += 1
                path = 'male/female'
        else:
            if p < 0.5:
                confusion_matrix[0,0] += 1
                path = 'female/female'
            else:
                confusion_matrix[0,1] += 1
                path = 'female/male'
        shutil.copy2(os.path.join(test_dir, file), os.path.join(base_dir, path))
    print confusion_matrix
    print 'accuracy:', np.trace(confusion_matrix)/np.sum(confusion_matrix)
    print 'female precision:', confusion_matrix[0,0]/np.sum(confusion_matrix[0,:])
    print 'male precision:', confusion_matrix[1,1]/np.sum(confusion_matrix[1,:])
    print 'female recall:', confusion_matrix[0,0]/np.sum(confusion_matrix[:,0])
    print 'male recall:', confusion_matrix[1,1]/np.sum(confusion_matrix[:,1])
