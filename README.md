# PreFab
`PreFab` (Prediction of Fabrication) is a package for modelling fabrication process induced variations in integrated photonic devices using deep convolutional neural networks.

Trained models can predict variations such as corner rounding (both over and under etching), washing away of small lines and islands, and filling of narrow holes and channels in a planar photonic structure. Once predicted, the designer can resimulate their design to rapidly prototype the expected performance and make any necessary changes prior to fabrication.

The package includes the tools used in the paper `Deep Learning Based Prediction of Fabrication-Process-Induced Structural Variations in Nanophotonic Devices` (which can be viewed here) for generating and processing training data, training predictor models, and using the predictor models on photonic designs. This package also includes pretrained models and the data they are trained on.

## Features
`pattern_generator.py` generates randomized patterns which can be fabricated and imaged to model a desired nanofabrication processes.

`dataset_compiler.py` processes the SEM and GDS images and fills out a dataset for model training and testing.

`model_trainer.py` trains the predictor model using TensorFlow.

`fabrication_predictor.py` predicts fabrication variations in a photonic design with the trained model(s).

Pretrained models are found in the `models/` directory.

An example dataset is found in the `datasets/` directory.

## Authors
`PreFab` was written by Dusan Gostimirovic with Odile Liboiron-Ladouceur, Danxia Xu, and Yuri Grinberg.

## License
This project is licensed under the terms of the MIT license.
