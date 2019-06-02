import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import h5py
import glob
import argparse
import os
import sys


class Annotator():
    def __init__(self, file_path):
        self.data = h5py.File(file_path, 'r+')
        self.points = []
        self.scatter_lines = []

        self.video = self.data['tissue/data'][:]
        self.fps = 1 / \
            (self.data['tissue/times'][5] - self.data['tissue/times'][4])

    def __del__(self):
        self.data.close()

    def ultraSoundAnimation(self):
        fig, ax = plt.subplots()
        line = ax.imshow(self.video[0, :, :], cmap='Greys_r')

        def init():
            line.set_data(self.video[0, :, :])
            return (line, )

        def animate(i):
            line.set_data(self.video[i, :, :])
            return (line, )

        def close(event):
            if event.key == 'Q':
                plt.close()

        interval = 1 / self.fps * 1000  # from framerate to ms
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=self.video.shape[0],
                                       interval=interval,
                                       blit=True)
        fig.canvas.mpl_connect('button_press_event', close)
        plt.show()

    def annotate(self):
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.ax.imshow(self.video[0, :, :], cmap='Greys_r')
        self.fig.canvas.mpl_connect('button_press_event', self.addPoint)
        self.fig.canvas.mpl_connect('key_press_event', self.keystrokeHandler)
        plt.show()

    def addPoint(self, event):
        point = [int(np.round(event.xdata)), int(np.round(event.ydata))]
        self.points.append(point)
        self.scatter_lines.append(self.ax.scatter(point[0], point[1],
                                                  color='red'))
        plt.draw()

    def keystrokeHandler(self, event):
        print(event.key)
        if event.key == 'backspace':
            self.removeLastPoint()
        elif event.key == ' ':
            self.savePoints()
        elif event.key == 'Q':
            plt.close()

    def removeLastPoint(self):
        self.points = self.points[:-1]
        self.scatter_lines[-1].remove()
        del self.scatter_lines[-1]
        plt.draw()

    def savePoints(self):
        p_array = np.array(self.points)
        try:
            self.data.create_dataset('tissue/track_points', data=p_array)
        except RuntimeError as e:
            if e.__str__() == "Unable to create link (name already exists)":
                del self.data['tissue/track_points']
                self.data['tissue/track_points'] = p_array
            else:
                raise e
        print("Saved points")


parser = argparse.ArgumentParser()

parser.add_argument('-dp', '--data-path')

args = parser.parse_args()

h5files = glob.glob(os.path.join(args.data_path, 'I599GE0C_1.h5'))
for h5file in h5files:
    print(h5file)
    annotator = Annotator(h5file)
    annotator.ultraSoundAnimation()

    annotator.annotate()

    del annotator
