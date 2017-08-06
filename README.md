# Gender Divide

The future is non-binary, but for now let's say it is to make things easier...

# Plan

1. Examine dataset by hand. Look for structure, mislabels, formats, etc.
2. Create image processing pipeline (resizing, downsampling, etc.).
3. Naive model implementation to use as baseline: Not-too-deep CNN w/ softmax.
4. Examine mislabeled items by hand.
5. Adjust for mislabelings (e.g. find person or not, and classify people on male/female).
6. Test data augmentation.
7. Avoid impulse to fine-tune model.
8. Make presentation.

# Requirements

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


Data is found at https://storage.cloud.google.com/fw-share/20170803-fw-gender-classification-data.zip. Assuming it is downloaded and in the root directory:

    unzip 20170803-fw-gender-classification-data.zip
    mv data-v2 data

Finally to reorganize data and create directory structure:

    python setup.py

# Dataset

## Description

Upon first look, the dataset consists of two directories of images (ones labelled male and female). The images are all square of varying resolution and contain anything for single clothing items to full scenes with people. Some images are clearly mislabeled (~ 1 in 30).

## Statistics

Counts are grabbed from the JSON metadata files. We found a single duplicate in the female training set. Other than that, all files are accounted for.

| Gender | Train | Test | Total |
| ------ | ----- | ---- | ----- |
| Female | 2664  | 370  | 3034  |
| Male   | 692   | 97   | 789   |

Resolutions are determined via Python's OpenCV wrapper. We plot them here:

### Female

![Female resolutions](results/female_resolutions.png)

### Male

![Male resolutions](results/male_resolutions.png)

# Models

To train:

    python train.py MODEL_NAME

To evaluate:

    python evaluate.py MODEL_NAME PATH_TO_HDF5_WEIGHTS

## NaiveCNN

|        | pred. female | pred. male |
| ------ | ------------ | ---------- |
| female | 294          | 76         |
| male   | 73           | 21         |

accuracy: 0.678879310345
female precision: 0.794594594595
male precision: 0.223404255319
female recall: 0.801089918256
male recall: 0.216494845361

## VGG16 (top layer only)

|        | pred. female | pred. male |
| ------ | ------------ | ---------- |
| female | 330          | 40         |
| male   | 40           | 57         |

accuracy: 0.82869379015
female precision: 0.891891891892
male precision: 0.587628865979
female recall: 0.891891891892
male recall: 0.587628865979

# Extensions

- pre-classification into people and objects
- retrain VGG16 layers
- classifier after feature selection
  - SVM
  - random forest
  - etc.
- ensembling
  - multiple models
  - multiple inputs from same input (via data augmentation)
- knobs
  - data augmentation (rotation, zoom, channel shifts, etc.)
  - dropout
  - regularization
  - resolution
  - learning rate (momentum, etc.)
- more data
  - amazon data
  - other web scraping (e.g. google images)
