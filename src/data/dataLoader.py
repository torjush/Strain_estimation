import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class DataLoader():
    """Takes care of getting images from h5files"""
    def __init__(self, data_path, shuffle=True):
        self.data_path = data_path
        self._findFiles()
        self.index_list = np.arange(0, len(self.h5files), dtype=int)
        if shuffle:
            np.random.shuffle(self.index_list)
        self.current_index = 0

    def _findFiles(self):
        h5files = [os.path.join(root, name)
                   for root, dirs, files in os.walk(self.data_path)
                   for name in files
                   if name.endswith(('.h5'))]
        if len(h5files) == 0:
            raise RuntimeError(f'Didn\'t find any h5files in directory: ' +
                               '{self.data_path}')
        self.h5files = h5files

    def loadSample(self):
        idx = self.index_list[self.current_index]
        with h5py.File(self.h5files[idx], 'r') as data:
            fixed = tf.constant(data['tissue/data'][:][:-1, :, :, None],
                                dtype='float32')
            moving = tf.constant(data['tissue/data'][:][1:, :, :, None],
                                 dtype='float32')

        # Comment out for overfitting experiment
        # if self.current_index == len(self.h5files) - 1:
        #     self.current_index = 0
        # else:
        #     self.current_index += 1
        return fixed, moving


if __name__ == '__main__':
    loader = DataLoader('../../data/processed/strain_point')
    fixed, moving = loader.loadSample()

    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 5))
    ax[0].imshow(fixed[0, :, :, 0], cmap='Greys_r')
    ax[1].imshow(moving[0, :, :, 0], cmap='Greys_r')
    plt.tight_layout()
    plt.show()
