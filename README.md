# PreFab
`PreFab` (Prediction of Fabrication) is used for modelling fabrication process induced variations in integrated photonic devices using deep convolutional neural networks.

Trained models predict variations such as corner rounding (both over and under etching), washing away of small lines and islands, and filling of narrow holes and channels in planar photonic structures. Once predicted, the designer resimulates their design to rapidly prototype the expected performance and make any necessary changes prior to (costly) fabrication.

This repository includes the tools used in the paper `Deep Learning Based Prediction of Fabrication-Process-Induced Structural Variations in Nanophotonic Devices` (which can be viewed here) for generating and processing training data, training the predictor model, and using the predictor model on photonic designs. This repository also includes a collection of data to train them on.

## Features
`pattern.py` generates randomized patterns which can be fabricated and imaged to model a desired nanofabrication process.

`dataset.py` processes the SEM and design images and fills out a dataset for model training and testing.

`trainer.py` trains the predictor model using TensorFlow.

`predictor.py` predicts fabrication variations in a photonic design with the trained model(s).

Matching SEM and design images are found in the `data/` directory.

Sample device images (to be predicted) are found in the `devices/` directory.

## Authors
`PreFab` was written by Dusan Gostimirovic with Odile Liboiron-Ladouceur, Danxia Xu, and Yuri Grinberg.

## Usage and Citation
For usage of `PreFab` in your work, please cite our journal article.

## License
This project is licensed under the terms of the MIT license. © 2022 National Research Council Canada and McGill University
