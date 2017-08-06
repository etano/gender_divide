import sys, shutil
from helpers import *
from models import NaiveModel

# Settings
batch_size = 16
img_width, img_height = 150, 150

# Predict
model = NaiveModel(tmp_dir, 'naive', img_width, img_height)
model.load(sys.argv[1])
predictions = model.predict(test_dir, batch_size)

# Copy files into directories to check
def makedirs(dirs):
    for dir in dirs:
        if not os.path.exists(os.path.join(tmp_dir, dir)):
            os.makedirs(os.path.join(tmp_dir, dir))
makedirs(['female/female', 'male/male', 'male/female', 'female/male'])
for [file, c, p] in predictions:
    print file, c, p
    print p >= 0.5
    if c == 1:
        if p >= 0.5:
            path = 'male/male'
        else:
            path = 'male/female'
    else:
        if p < 0.5:
            path = 'female/female'
        else:
            path = 'female/male'
    shutil.copy2(os.path.join(test_dir, file), os.path.join(tmp_dir, path))
