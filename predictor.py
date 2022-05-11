import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class Predictor():
    def __init__(self, run, version, model_nums):
        self.run = run
        self.version = version
        self.model_nums = model_nums
        self.models = []
        for j in self.model_nums:
            self.models.append(models.load_model("models/predictor_"
                + self.run + "_v" + str(self.version) + "_m" + str(j) + ".pb"))
        self.slice_size = int(np.sqrt(self.models[0].weights[-1].shape)[0])

    def predict(self, device, step_size, binarize):
        # cut up the device
        slice_size = self.slice_size
        xs = ys = 0
        xe = ye = slice_size
        device_images = []
        while xs < device.shape[1]:
            ys = 0
            ye = slice_size
            while ys < device.shape[0]:
                image = np.zeros((slice_size, slice_size))
                image[0:(ye-ys), 0:(xe-xs)] = device[ys:ye, xs: xe]
                device_images.append(image)
                ys += step_size
                if ye + step_size > device.shape[0]:
                    ye = device.shape[0]
                else:
                    ye += step_size
            xs += step_size
            if xe + step_size > device.shape[1]:
                xe = device.shape[1]
            else:
                xe += step_size

        # inference the device images
        y_sum = 0
        for model in self.models:
            x = tf.convert_to_tensor(device_images, dtype = tf.float32)
            y = np.squeeze(model(x).numpy())
            y_sum += y
        y = y_sum/len(self.model_nums)

        # stitch device back together
        prediction = np.zeros_like(device)
        avg_matrix = np.zeros_like(device)
        xs = ys = 0
        xe = ye = slice_size
        ind = 0
        while xs < prediction.shape[1]:
            ys = 0
            ye = slice_size
            while ys < prediction.shape[0]:
                prediction[ys:ye, xs:xe] += y[ind][0:(ye-ys), 0:(xe-xs)]
                avg_matrix[ys:ye, xs:xe] += np.ones(((ye-ys), (xe-xs)))
                ys += step_size
                if ye + step_size > device.shape[0]:
                    ye = device.shape[0]
                else:
                    ye += step_size
                ind += 1
            xs += step_size
            if xe + step_size > device.shape[1]:
                xe = device.shape[1]
            else:
                xe += step_size

        # averaging the overlaps
        prediction = prediction/avg_matrix

        # binarize
        if binarize:
            prediction = cv2.GaussianBlur(prediction, (5, 5), 0)
            prediction[prediction >= 0.5] = 1
            prediction[prediction < 0.5] = 0

        return prediction

# %% codecell
# load a device image in and prepare for prediction
device = img.imread('devices/demux.png')[:, :, 1]
res = 1.655 # resolution of model's training data (px/nm)
length_nm = 5000 # length of device (nm)
scale = 1/(res/(length_nm/device.shape[1]))
device = cv2.resize(device, (0, 0), fx = scale, fy = scale)
device[device < 0.5], device[device >= 0.5] = 0, 1
device = np.pad(device, [(100, 100), (100, 100)], mode = 'constant')
device_size = (res*device.shape[0], res*device.shape[1])

# run the prediction
run = "example"     # name of the data directory (of dataset)
version = 0         # version number (user defined, of dataset)
model_nums = [0]    # can list multiple models here for smoother prediction
step_size = 32      # step size for prediction scan (px)
p = Predictor(run = run, version = version, model_nums = model_nums)
prediction = p.predict(device = device, step_size = step_size, binarize = False)

# show the results
plt.imshow(device, extent = [-device_size[1]/2, device_size[1]/2,
    -device_size[0]/2, device_size[0]/2])
plt.title('Original Device')
plt.ylabel('Distance (nm)')
plt.xlabel('Distance (nm)')
plt.show()

plt.imshow(prediction, extent = [-device_size[1]/2, device_size[1]/2,
    -device_size[0]/2, device_size[0]/2])
plt.title('Predicted Device')
plt.ylabel('Distance (nm)')
plt.xlabel('Distance (nm)')
plt.show()
