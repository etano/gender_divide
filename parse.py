"""Parses Amazon product metadata found at http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz"""

import sys, os, yaml, csv, urllib
from yaml import CLoader as Loader

def usage():
    print """
USAGE: python parse.py metadata.json target_dir
"""
    sys.exit(0)

def main(argv):
    if len(argv) < 3:
        usage()
    filename = sys.argv[1]
    target_dir = sys.argv[2]
    with open(filename, 'rb') as f:
        products = []
        count, good, bad = 0, 0, 0
        for line in f:
            count += 1
            if not (count % 100):
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
                    if not os.path.exists(file):
                        urllib.urlretrieve(imUrl, file)
                    good += 1
                except Exception as e:
                    print line
                    print e
                    bad += 1
        print "good:", good, ", bad:", bad

if __name__ == "__main__":
    main(sys.argv)
