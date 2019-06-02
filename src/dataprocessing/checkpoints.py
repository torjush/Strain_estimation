import h5py
import glob
import argparse
import os
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--data-path')

args = parser.parse_args()


def xwin(event):
    if event.key == 'Q':
        plt.close()


h5files = glob.glob(os.path.join(args.data_path, '*.h5'))
for h5file in h5files:
    print(h5file)
    with h5py.File(h5file, 'r') as data:
        plt.imshow(data['tissue/data'][0, :, :], cmap='Greys_r')

        points = data['tissue/track_points']
        for i in range(points.shape[0]):
            plt.scatter(points[i, 0], points[i, 1])

        plt.gcf().canvas.mpl_connect('key_press_event', xwin)
        plt.show()
