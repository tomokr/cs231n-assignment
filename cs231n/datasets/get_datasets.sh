#!/bin/bash
# Get CIFAR10
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*) wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz;;
    Darwin*) curl -O http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz;;
esac

tar -xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz 
