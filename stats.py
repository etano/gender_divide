import os, json, cv2
import numpy as np
import matplotlib.pyplot as plt

# Global
img_dir = './data/img'
meta_dir = './data/meta'

# Count files
n_files = lambda x: len([f for f in os.listdir(x) if os.path.isfile(os.path.join(x, f))])
print 'File count: '
print '  Female: %d' % n_files(os.path.join(img_dir, 'female'))
print '  Male: %d' % n_files(os.path.join(img_dir, 'male'))

# Get metadata (removes duplicates)
def get_gender_metadata(file):
    with open(file) as x:
        data = json.load(x)
        return [os.path.join(img_dir, f) for f in set(data['female'])], [os.path.join(img_dir, f) for f in set(data['male'])]
female_train, male_train = get_gender_metadata(os.path.join(meta_dir, 'train.json'))
female_test, male_test = get_gender_metadata(os.path.join(meta_dir, 'test.json'))
female_all = female_train + female_test
male_all = male_train + male_test
print 'Metadata count: '
print '  Female: %d (train: %d, test: %d)' % (len(female_all), len(female_train), len(female_test))
print '  Male: %d (train: %d, test: %d)' % (len(male_all), len(male_train), len(male_test))

# Check all files exist
n_bad_files = lambda x: len([f for f in x if (not os.path.isfile(f))])
print 'File check: '
print '  Female: %d non-existent files' % (n_bad_files(female_all))
print '  Male: %d non-existent files' % (n_bad_files(male_all))

# Plot resolutions
def get_resolution_counts(files):
    res_dict = {}
    for file in files:
        res = cv2.imread(file).shape
        if not (res in res_dict): res_dict[res] = 1
        else: res_dict[res] += 1
    return res_dict
def plot_resolutions(res_dict, name):
    heights, widths, counts = [], [] ,[]
    for [height, width, channels], count in res_dict.iteritems():
        heights.append(height)
        widths.append(width)
        counts.append(count)
    plt.scatter(heights, widths, c=counts)
    plt.gray()
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,x2,0,y2))
    plt.xlabel('height')
    plt.ylabel('width')
    plt.savefig(name)
    plt.clf()
print 'Plotting resolutions: '
print '  Female...'
plot_resolutions(get_resolution_counts(female_all), 'female_resolutions.png')
print '  Male...'
plot_resolutions(get_resolution_counts(male_all), 'male_resolutions.png')


print 'Done.'
