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

    def predict(self, device, step_size):
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
        for j in self.model_nums:
            model = self.models[j]
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
        prediction = cv2.GaussianBlur(prediction, (5, 5), 0)
        prediction[prediction >= 0.5] = 1
        prediction[prediction < 0.5] = 0

        return prediction

# %% codecell
# sample usage
device = img.imread('devices/box.png')[:, :, 1]
res = 1.6548463356973995
device_size = (res*device.shape[0], res*device.shape[1])

model_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
p = Predictor(run = 'example', version = 0, model_nums = model_nums)
prediction = p.predict(device = device, step_size = 16)
plt.imshow(prediction, extent = [-device_size[0]/2, device_size[0]/2,
    -device_size[1]/2, device_size[1]/2])
plt.title('Prediction')
plt.ylabel('Distance (nm)')
plt.xlabel('Distance (nm)')
plt.show()
