"""Parses Amazon product metadata found at http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz"""

import sys, os, yaml, csv, urllib, tqdm, gzip
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from multiprocessing import Pool

def parse_metadata(filename):
    lines = []
    with gzip.open(filename, 'rb') as f:
        lines = f.read().splitlines()
    return lines

def parse_line(line):
    if ("'imUrl':" in line) and ("'categories':" in line):
        try:
            line = line.rstrip().replace("\\'","''")
            product = yaml.load(line, Loader=Loader)
            url, categories = product['imUrl'], product['categories'][0]
            subcategory = categories[1] if len(categories) > 1 else None
            if subcategory == 'Men':
                gender = 'male'
            elif subcategory == 'Women':
                gender = 'female'
            else:
                return
            filename = url.split('/')[-1]
            download_file(url, os.path.join(target_dir, gender+'/'+filename))
        except Exception as e:
            return

def download_file(url, dst):
    if not os.path.exists(dst):
        urllib.urlretrieve(url, dst)

n_threads = 20
target_dir = "./data/amazon"
filename = 'meta_Clothing_Shoes_and_Jewelry.json.gz'
amazon_url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'+filename

print 'Downloading metadata...'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
download_file(amazon_url, os.path.join(target_dir, filename))

print 'Loading metadata...'
lines = parse_metadata(os.path.join(target_dir, filename))

print 'Downloading images...'
if not os.path.exists(os.path.join(target_dir, 'female')):
    os.makedirs(os.path.join(target_dir, 'female'))
if not os.path.exists(os.path.join(target_dir, 'male')):
    os.makedirs(os.path.join(target_dir, 'male'))
pool = Pool(n_threads)
for _ in tqdm.tqdm(pool.imap_unordered(parse_line, lines), total=len(lines)):
    pass
