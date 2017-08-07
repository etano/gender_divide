import sys, os, json

img_dir = sys.argv[1]
filename = sys.argv[2]

female_imgs = os.listdir(os.path.join(img_dir, 'female'))
male_imgs = os.listdir(os.path.join(img_dir, 'male'))
meta = {
    'female': ['female/'+img for img in female_imgs],
    'male': ['male/'+img for img in male_imgs]
}

dir = '/'.join(filename.split('/')[:-1])
if not os.path.exists(dir):
    os.makedirs(dir)
with open(filename, 'w') as f:
    json.dump(meta, f, indent=4)
