[Русский 👈](./README_ru.md)
# nn-practice
This project is meant to be a practical test of my knowledge of classical neural networks. It is implemented using NumPy. A notable highlight of this implementation is that this model supports vector activation functions that output values that are dependent on all inputs simultaneously (e.g. Softmax).

## Requirements
Neural network implementation has following dependencies:
* Python 3.6 and above
* NumPy package

## Demo
This repository includes a Jupyter notebook with a demo of this model. It runs training and classification on MNIST handwritted digits dataset.
The demo further depends on packages `pickle` and `gzip`.

## Credits
Model user-facing interface, general structure and documentation style are loosely adapted from https://github.com/mnielsen/neural-networks-and-deep-learning. Files `mnist.pkl.gz` and `mnist_loader_mod.py` for the demo are taken directly from this repository, with the latter modified to work with Python 3.
