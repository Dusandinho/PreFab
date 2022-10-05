import matplotlib.image as img
import numpy as np
import cv2

def binarize(device, eta=0.5, beta=np.inf):
    num = np.tanh(beta*eta) + np.tanh(beta*(device - eta))
    den = np.tanh(beta*eta) + np.tanh(beta*(1 - eta))
    device = num/den
    return device

def binarize_hard(device, eta=0.5):
    device[device < eta] = 0
    device[device >= eta] = 1
    return device

def trim(device):
    x, y = np.nonzero(device)
    xl, xr = x.min(), x.max()
    yl, yr = y.min(), y.max()
    return device[xl:xr+1, yl:yr+1]

def pad(device, slice_length, fctr=1):
    pady = (slice_length*np.ceil(device.shape[0]/slice_length) - device.shape[0])/2 + slice_length*(fctr - 1)/2
    padx = (slice_length*np.ceil(device.shape[1]/slice_length) - device.shape[1])/2 + slice_length*(fctr - 1)/2
    device = np.pad(device, [(int(np.ceil(pady)), int(np.floor(pady))),
        (int(np.ceil(padx)), int(np.floor(padx)))], mode='constant')
    return device

def load_device(path, slice_length, device_length, res):
    device = img.imread(path)[:,:,1]

    # scale the device so that res*length_px = length_nm
    # units are still px, but different
    device = trim(device)
    scale = 1/(res/(device_length/device.shape[1]))
    device = cv2.resize(device, (0,0), fx=scale, fy=scale)
    device = binarize(device)

    # pad to multiple of slice_length
    device = pad(device, slice_length=slice_length, fctr=2)
    return device
