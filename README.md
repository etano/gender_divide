# Gender Divide

The future is non-binary, but for now let's say it is to make things easier...

# Installation

Assuming Ubuntu and root priviledges:

    sudo apt-get update && sudo apt-get install -y python-pip zip hdf5-tools
    sudo pip install --upgrade pip
    sudo pip install keras
    sudo pip install tensorflow
    sudo pip install opencv-python
    sudo pip install Pillow
    sudo pip install h5py
    sudo pip install imutils
    sudo pip install tqdm

# Data

## Amazon

Image data from Amazon. It can be downloaded with download_amazon_images.py. It will take about an hour.

Then setup with:

    mkdir -p data
    mv amazon.tar.gz data/
    tar -xzvf amazon.tar.gz
    cd ..
    python make_meta_json.py data/amazon/img data/amazon/meta 0.1 15000
    python setup.py data/amazon

where 0.1 is the test fraction and 15000 is the maximum number of images

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
