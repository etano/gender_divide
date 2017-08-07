import sys, os, json, random

img_dir = sys.argv[1]
filename = sys.argv[2]
max_imgs = int(sys.argv[3])

female_imgs = os.listdir(os.path.join(img_dir, 'female'))
male_imgs = os.listdir(os.path.join(img_dir, 'male'))
random.shuffle(female_imgs)
random.shuffle(male_imgs)

meta = {'female': [], 'male': []}

for i in range(max_imgs):
    meta['female'].append('female/'+female_imgs[i])
    meta['male'].append('male/'+male_imgs[i])

dir = '/'.join(filename.split('/')[:-1])
if not os.path.exists(dir):
    os.makedirs(dir)
with open(filename, 'w') as f:
    json.dump(meta, f, indent=4)
