# PreFab
`PreFab` (Prediction of Fabrication) is a package for modelling fabrication process induced variations in integrated photonic devices with deep convolutional neural networks.

Trained models can predict variations like corner rounding, washing away of small holes/features, and general over/under-etching in any planar photonic structure. Once predicted, the designer can resimulate their design to rapidly prototype the expected performance and make any necessary changes.

The package includes the tools used in the paper `Deep Learning Based Prediction of Fabrication-Process-Induced Structural Variations in Nanophotonic Devices` (which can be viewed here) for generating and processing training data, training predictor models, and using the predictor models on photonic designs. It also includes pretrained models and the data they're trained on.

## Features
`pattern_generator.py` generates randomized patterns. These patterns can be fabricated and imaged to model a desired nanofabrication processes.

`dataset_generator.py` processes the SEM and GDS images and fills out a datatset for model training and testing.

`model_trainer.py` trains the predictor model using PyTorch.

`fabrication_predictor.py` predicts fabrication variation in a photonic design with the trained model(s).

Pretrained models are found in the `models/` directory.

An example dataset are found in the `datasets/` directory.

## Authors
`PreFab` was written by Dusan Gostimirovic with Odile Liboiron-Ladouceur, Danxia Xu, and Yuri Grinberg.

## License
This project is licensed under the terms of the MIT license.
