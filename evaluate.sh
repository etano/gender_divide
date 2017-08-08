#!/bin/bash

#python evaluate.py naive_cnn_data_v2 naive_cnn data/data_v2/ results/naive_cnn_data_v2.h5 > results/naive_cnn_data_v2.dat
#python evaluate.py naive_cnn_corrected naive_cnn data/corrected/ results/naive_cnn_corrected.h5 > results/naive_cnn_corrected.dat
#python evaluate.py naive_cnn_amazon naive_cnn data/amazon/ results/naive_cnn_amazon.h5 > results/naive_cnn_amazon.dat
#
#python evaluate.py vgg16_data_v2 vgg16 data/data_v2/ results/vgg16_data_v2.h5 > results/vgg16_data_v2.dat
#python evaluate.py vgg16_corrected vgg16 data/corrected/ results/vgg16_corrected.h5 > results/vgg16_corrected.dat
#python evaluate.py vgg16_amazon vgg16 data/amazon/ results/vgg16_amazon.h5 > results/vgg16_amazon.dat

python evaluate.py face_gender_data_v2 face_gender data/data_v2 > results/face_gender_data_v2.dat
#python evaluate.py face_gender_corrected face_gender data/corrected > results/face_gender_corrected.dat
#
#python evaluate.py ensemble ensemble data/corrected > results/ensemble.dat
