#!/bin/bash

python evaluate.py naive_cnn_amazon naive_cnn data/amazon/ results/naive_cnn_amazon.h5 > results/naive_cnn_amazon.dat
python evaluate.py vgg16_amazon vgg16 data/amazon/ results/vgg16_amazon.h5 > results/vgg16_amazon.dat
python evaluate.py face_gender_amazon face_gender data/amazon > results/face_gender_amazon.dat
python evaluate.py ensemble ensemble data/amazon > results/ensemble.dat
