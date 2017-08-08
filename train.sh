#!/bin/bash

time python train.py naive_cnn_data_v2 naive_cnn data/data_v2
time python train.py naive_cnn_corrected naive_cnn data/corrected
time python train.py naive_cnn_amazon naive_cnn data/amazon

time python train.py vgg16_data_v2 vgg16 data/data_v2
time python train.py vgg16_corrected vgg16 data/corrected
time python train.py vgg16_amazon vgg16 data/amazon
