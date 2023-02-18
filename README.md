## Preambula

This repository contains code to train the image classification model using the semi-supervised approach known as [Unsupervised Data Augmentation](https://arxiv.org/abs/1904.12848)

I ported the [initial implementation](https://github.com/google-research/uda) for images in Tensorflow to PyTorch, so this trainer uses:
- RandAugment to augment image data (using albumentations to improve the performance)
- Confidence-based masking
- Training Signal Annealing
- Predictions Sharpening using Softmax temperature

The code was primarily developed for the [Wildlife Image Classification Competition](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/), but may be pretty easily adapted to any other image classification task.