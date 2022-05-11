# PreFab
`PreFab` (Prediction of Fabrication) is used for modelling fabrication process induced variations in integrated photonic devices using deep convolutional neural networks.

Trained models predict variations such as corner rounding (both over and under etching), washing away of small lines and islands, and filling of narrow holes and channels in planar photonic structures. Once predicted, the designer resimulates their design to rapidly prototype the expected performance and make any necessary changes prior to (costly) fabrication.

This repository includes the tools used in the paper `Deep Learning Based Prediction of Fabrication-Process-Induced Structural Variations in Nanophotonic Devices` (which can be viewed here) for generating and processing training data, training the predictor model, and using the predictor model on photonic designs. This repository also includes a collection of data to train them on.

![promo](images/promo.png)
<figcaption align = "center">Figure 1 - Prediction of fabrication variation in a simple structure on a 220 nm silicon-on-insulator electron-beam lithography process. Prediction time of 8.2 seconds.</figcaption>

## Features
`pattern.py` generates randomized patterns which can be fabricated and imaged to model a desired nanofabrication process.

`dataset.py` processes the SEM and design images and fills out a dataset for model training and testing.

`trainer.py` trains the predictor model using TensorFlow.

`predictor.py` predicts fabrication variations in a photonic design with the trained model(s).

Matching SEM and design images are found in the `data/` directory.

Sample device images (to be predicted) are found in the `devices/` directory.

## Notes and Examples
`PreFab` can be used in the following order to get started:
1. Run `pattern.py` to generate design patterns for fabrication characterization
    - Adjust pattern and filter sizes to suit your needs
    - Generate a small collection of patterns to add to your layout (for fabrication and imaging) so that they can be used in training a model of your nanofabrication process


2. Run `dataset.py` to prepare your data for training:
    - See `data/example/` for a simple example of how GDS and SEM data should be arranged prior to running this script. Feel free to use the example data to get familiar with the process
    - This script cuts the GDS and SEM images into smaller, more manageable slices for training and prediction. Adjust the slice and scanning step sizes to suit your needs


3. Run `train.py` to train a model on your fabrication dataset:
    - This script uses [TensorFlow](https://www.tensorflow.org/install) to handle the training
    - Feel free to adjust the structure of the model and its hyperparameters to try to improve the prediction accuracy


4. Run `predictor.py` to predict the fabrication variations of your device:
    - A topologically optimized WDM DEMUX and some simple shapes are included in `devices/` to help in getting familiar with the process
    - To use your own device design, the script must know the length (nm) and the resolution (px/nm) of the model's training data
    - For high-accuracy prediction, use multiple models and a small scanning step size

## Authors
`PreFab` was written by Dusan Gostimirovic with Odile Liboiron-Ladouceur, Danxia Xu, and Yuri Grinberg.

## Usage and Citation
For usage of `PreFab` in your work, please cite our journal article.

## License
This project is licensed under the terms of the MIT license. © 2022 National Research Council Canada and McGill University.
