# Gender Divide

The future is non-binary, but for now let's say it is to make things easier...

# Installation

Assuming Ubuntu and root priviledges:

    sudo apt-get install python-pip zip hdf5-tools
    sudo pip install --upgrade pip
    sudo pip install keras
    sudo pip install tensorflow
    sudo pip install opencv-python
    sudo pip install Pillow
    sudo pip install h5py
    sudo pip install imutils
    sudo pip install tqdm

# Data

## Fashwell

To download use https://storage.cloud.google.com/fw-share/20170803-fw-gender-classification-data.zip

Then setup with:

    mkdir -p data
    mv 20170803-fw-gender-classification-data.zip data/
    cd data/
    unzip 20170803-fw-gender-classification-data.zip
    cd ..
    python setup.py data/data_v2

## Corrected

Some images are clearly mislabeled. These were fixed by hand and can be downloaded at https://drive.google.com/uc?export=download&confirm=iD8U&id=0B62SS8vA5tqgYnNoYWw0NHZvV1U

Then setup with:

    mkdir -p data
    mv corrected.tar.gz data/
    tar -xzvf corrected.tar.gz
    cd ..
    python make_meta_json.py data/corrected/img data/corrected/meta data/data_v2/meta/test.json
    python setup.py data/corrected

## Amazon

Data was supplemented with data from Amazon. It can be downloaded at https://drive.google.com/open?id=0B62SS8vA5tqgMlVYWklMWmpoaEk

Then setup with:

    mkdir -p data
    mv amazon.tar.gz data/
    tar -xzvf amazon.tar.gz
    cd ..
    python make_meta_json.py data/amazon/img data/amazon/meta data/data_v2/meta/test.json 15000
    python setup.py data/amazon

# Train

NOTE: Pretrained models already exist in the `results` folder.

To train all models (this will take hours), run:

    ./train.sh

To train individual models, run:

    python train.py NAME MODEL_TYPE DATA_DIR (CHECKPOINT_HDF5_FILE)

# Evaluate

To evaluate all models (this will take minutes), run:

    ./evaluate.sh

To evaluate individual models, run:

    python evaluate.py NAME MODEL_TYPE DATA_DIR (CHECKPOINT_HDF5_FILE)
