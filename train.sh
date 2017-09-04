#!/bin/bash

time python train.py naive_cnn_amazon naive_cnn data/amazon
time python train.py vgg16_amazon vgg16 data/amazon
