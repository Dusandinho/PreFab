# NN-PFP
`NN-PFP` (Neural Network Photonic Fabrication Predictor) is a package for modelling fabrication process induced variations in integrated photonic devices with deep convolutional neural networks.

Trained models can predict variations like corner rounding, washing away of small holes/features, and general over/under-etching in any planar photonic structure. Once predicted, the designer can resimulate their design to rapidly prototype the expected performance.

The package includes the tools used in the paper `Deep Learning Based Prediction of Fabrication-Process-Induced Structural Variations in Nanophotonic Devices` (which can be viewed here), for generating and processing training data, training predictor models, and using the predictor models on photonic designs. It also includes pretrained models and the data they're trained on.

# What's Included
`dataset_generator.py` processes the SEM and GDS images and creates the training datatset.

`model_trainer.py` trains the predictor model using PyTorch.

Pretrained models are found in the `models/` directory.

# Authors
`NN-PFP` was written by Dusan Gostimirovic with Odile Liboiron-Ladouceur, Danxia Xu, and Yuri Grinberg.
