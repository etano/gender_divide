#!/bin/bash

time python train.py vgg16_data_v2 vgg16 data/data_v2
time python train.py vgg16_corrected vgg16 data/corrected
time python train.py vgg16_amazon vgg16 data/amazon