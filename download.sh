#!/bin/bash

export DATA_DIR=.

mkdir -p $DATA_DIR/mnist

echo "$DATA_DIR/" > data_dir.txt

wget -O $DATA_DIR/mnist/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -O $DATA_DIR/mnist/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -O $DATA_DIR/mnist/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -O $DATA_DIR/mnist/t10k-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip $DATA_DIR/mnist/train-images-idx3-ubyte.gz
gunzip $DATA_DIR/mnist/train-labels-idx1-ubyte.gz
gunzip $DATA_DIR/mnist/t10k-images-idx3-ubyte.gz
gunzip $DATA_DIR/mnist/t10k-labels-idx1-ubyte.gz
