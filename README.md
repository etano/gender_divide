# Gender Divide

The future is non-binary, but for now let's say it is to make things easier...

# Requirements

- Python 2.7+
- Keras
- Tensorflow
- OpenCV

# Installation

Assuming Ubuntu and root priviledges:

    apt-get install python-pip zip hdf5
    pip install keras
    pip install tensorflow
    pip install opencv-python
    pip install h5py
    pip install imutils
    pip install tqdm

# Data

## Fashwell

To download and setup the original dataset:

    mkdir -p data && cd data
    wget https://storage.cloud.google.com/fw-share/20170803-fw-gender-classification-data.zip
    unzip 20170803-fw-gender-classification-data.zip
    cd ..
    python setup.py data/data_v2

## Corrected

Some images are clearly mislabeled. These were fixed by hand and can be downloaded and setup as follows:

    mkdir -p data && cd data
    wget https://www.dropbox.com/s/w7v47ssa63oml1i/corrected.tar.gz
    tar -xzvf corrected.tar.gz
    cd ..
    python make_meta_json data/amazon/img data/amazon/meta data/data_v2/meta/test.json
    python setup.py data/corrected

## Amazon

Data was supplemented with data from Amazon. It can be downloaded and setup as follows:

    mkdir -p data && cd data
    wget https://www.dropbox.com/s/q33ftmphiofnv4m/amazon.tar.gz
    tar -xzvf amazon.tar.gz
    cd ..
    python make_meta_json data/amazon/img data/amazon/meta data/data_v2/meta/test.json 15000
    python setup.py data/amazon

# Train

To train all models (this will take hours), run:

    ./train.sh

To train individual models, run:

    python train.py NAME MODEL_TYPE DATA_DIR (CHECKPOINT_HDF5_FILE)

# Evaluate

To evaluate all models (this will take minutes), run:

    ./evaluate.sh

To evaluate individual models, run:

    python evaluate.py NAME MODEL_TYPE DATA_DIR (CHECKPOINT_HDF5_FILE)
