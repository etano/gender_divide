import argparse, cv2
import numpy as np
from imutils import paths

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
ap.add_argument("-t", "--target_dir", required=True, help="path to target directory")
args = vars(ap.parse_args())

# loop over the image paths
cc = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
for image_path in paths.list_images(args["images"]):
    rgb_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    faces = cc.detectMultiScale(gray_image, 1.3, 5)
    count = 0
    filename = image_path.split('/')[-1]
    for (x, y, width, height) in faces:
        path = args["target_dir"]+'/'+str(count)+'_'+filename
        cv2.imwrite(path, rgb_image[y:y+height, x:x+width])
        count += 1
