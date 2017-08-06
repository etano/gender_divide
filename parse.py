"""Parses Amazon product metadata found at http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz"""

import sys, os, yaml, csv, urllib, tqdm
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from multiprocessing import Pool

def usage():
    print """
USAGE: python parse.py metadata.json target_dir
"""
    sys.exit(0)

def get_files_to_download(filename, target_dir):
    files = []
    with open(filename, 'rb') as f:
        count, good, bad = 0, 0, 0
        for line in f:
            count += 1
            if not (count % 10000):
                print "count:", count, "good:", good, ", bad:", bad
            if ("'imUrl':" in line) and ("'categories':" in line):
                try:
                    line = line.rstrip().replace("\\'","''")
                    product = yaml.load(line, Loader=Loader)
                    imUrl, categories = product['imUrl'], product['categories'][0]
                    subcategory = categories[1] if len(categories) > 1 else None
                    if subcategory == 'Men':
                        gender = 'male'
                    elif subcategory == 'Women':
                        gender = 'female'
                    else:
                        continue
                    file = os.path.join(target_dir, gender+'/'+imUrl.split('/')[-1])
                    files.append([imUrl, file])
                    good += 1
                except Exception as e:
                    print line
                    print e
                    bad += 1
    return files

def download_file(file):
    [url, dst] = file
    try:
        if not os.path.exists(dst):
            urllib.urlretrieve(url, dst)
    except Exception as e:
        print url, dst

if __name__ == "__main__":
    if len(sys.argv) < 3:
        usage()
    filename = sys.argv[1]
    target_dir = sys.argv[2]
    files = get_files_to_download(filename, target_dir)

    pool = Pool(8)
    for _ in tqdm.tqdm(pool.imap_unordered(download_file, files), total=len(files)):
        pass
