import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fft
from scipy.io import savemat

class Pattern():
    def __init__(self, height, width, fctr):
        self.height = height
        self.width = width
        self.fctr = fctr
        self.pattern = self._generate_pattern()

    def show_pattern(self):
        plt.title('Randomly Generated Pattern')
        plt.ylabel('Distance (px, nm)')
        plt.xlabel('Distance (px, nm)')
        plt.imshow(self.pattern)
        plt.show()

    def export_mat4gds(self, filename):
        # this can be used in a .mat to .gds converter
        savemat(filename, {"pattern": self.pattern,
            "period": [self.width, self.height]})

    def _generate_pattern(self):
        self.f = np.random.uniform(low = 0, high = 1,
            size = (self.height, self.width))
        self._transform()
        self._reduce_transform()
        self._inv_transform()
        self.f = self.f.real
        self.f[self.f < 0.5] = 0
        self.f[self.f >= 1] = 1

        return self.f

    def _transform(self):
        self.f = scipy.fft.fft2(self.f)
        self.f = scipy.fft.fftshift(self.f)

    def _reduce_transform(self):
        filter_margin = int((self.width - self.width*self.fctr)/2)
        rx = range(filter_margin, self.width - filter_margin)
        ry = range(filter_margin, self.height - filter_margin)
        filter = np.zeros_like(self.f)
        filter[ry[0]:ry[-1], rx[0]:rx[-1]] = 1
        self.f *= filter

    def _inv_transform(self):
        self.f = scipy.fft.ifftshift(self.f)
        self.f = scipy.fft.ifft2(self.f)

# %% codecell
# sample usage
p = Pattern(height = 3000, width = 3000, fctr = 0.01)
p.show_pattern()

# p.export_mat4gds('random-pattern_' + str(p.height) + 'x' + str(p.width) + '_'
#     + str(p.fctr) + '.mat')
