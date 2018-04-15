#!/bin/bash

mkdir -p ./data/mnist

if ! [ -e ./data/mnist/train-images-idx3-ubyte.gz ]
	then
		wget -P ./data/mnist/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
fi
gzip -d ./data/mnist/train-images-idx3-ubyte.gz

if ! [ -e ./data/mnist/train-labels-idx1-ubyte.gz ]
	then
		wget -P ./data/mnist/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
fi
gzip -d ./data/mnist/train-labels-idx1-ubyte.gz

if ! [ -e ./data/mnist/t10k-images-idx3-ubyte.gz ]
	then
		wget -P ./data/mnist/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
fi
gzip -d ./data/mnist/t10k-images-idx3-ubyte.gz

if ! [ -e ./data/mnist/t10k-labels-idx1-ubyte.gz ]
	then
		wget -P ./data/mnist/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
fi
gzip -d ./data/mnist/t10k-labels-idx1-ubyte.gz
