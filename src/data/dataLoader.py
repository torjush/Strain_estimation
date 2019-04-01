import h5py
import numpy as np
import matplotlib.pyplot as plt


class DataSet():
    def __init__(self, data_path):
        self.__dataset = h5py.File(data_path, 'r')
        self.fixed = self.__dataset['fixed']
        self.moving = self.__dataset['moving']

        self.num_samples = self.fixed.shape[0]

    # def __del__(self):
    #     self.__dataset.close()

    def batchGenerator(self, batch_size, shuffle=True):
        idc = list(range(self.num_samples))
        if shuffle:
            np.random.shuffle(idc)

        for i in range(self.num_samples // batch_size):
            from_idx = i * batch_size
            to_idx = from_idx + batch_size
            sample_idc = sorted(idc[from_idx:to_idx])
            yield (self.fixed[sample_idc, :, :, :],
                   self.moving[sample_idc, :, :, :])

        # Take care of last run
        if self.num_samples % batch_size:
            sample_idc = sorted(idc[(i + 1) * batch_size:])
            yield (self.fixed[sample_idc, :, :, :],
                   self.moving[sample_idc, :, :, :])


if __name__ == '__main__':
    loader = DataSet('../../data/processed/strain_est_low_fps/train.h5')
    batch_gen = loader.batch_generator(16)

    for fixed, moving in batch_gen:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        ax[0][0].imshow(fixed[4, :, :, 0], cmap='Greys_r')
        ax[0][0].set_title('Fixed 4')
        ax[0][1].imshow(moving[4, :, :, 0], cmap='Greys_r')
        ax[0][1].set_title('Moving 4')
        ax[1][0].imshow(fixed[-1, :, :, 0], cmap='Greys_r')
        ax[1][0].set_title('Fixed -1')
        ax[1][1].imshow(moving[-1, :, :, 0], cmap='Greys_r')
        ax[1][1].set_title('Moving -1')

        plt.tight_layout()
        plt.show()
