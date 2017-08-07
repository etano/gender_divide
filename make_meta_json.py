import sys, os, json, random

img_dir = sys.argv[1]
target_dir = sys.argv[2]
test_meta = json.load(open(sys.argv[3], 'r'))
max_imgs = int(sys.argv[4]) if len(sys.argv) > 4 else None

def get_imgs(gender, dir):
    imgs = []
    for img in os.listdir(os.path.join(dir, gender)):
        if not (gender+'/'+img in test_meta[gender]):
            imgs.append(gender+'/'+img)
    if max_imgs == None:
        return imgs
    else:
        random.shuffle(imgs)
        return imgs[:max_imgs]

train_meta = {
    'female': get_imgs('female', img_dir),
    'male': get_imgs('male', img_dir)
}

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
with open(os.path.join(target_dir, 'train.json'), 'w') as f:
    json.dump(train_meta, f, indent=4)
