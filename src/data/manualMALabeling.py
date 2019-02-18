import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob


class MouseCross():
    def __init__(self, ax, **kwargs):
        self.ax = ax
        self.line, = self.ax.plot([0], [0], visible=False, **kwargs)

    def show_cross(self, event):
        if event.inaxes == self.ax:
            self.line.set_data([event.xdata], [event.ydata])
            self.line.set_visible(True)
        else:
            self.line.set_visible(False)
        plt.draw()

    def place_cross(self, event):
        self.ax.scatter(event.xdata, event.ydata, marker='x', color='red')


class Backend():
    def __init__(self, path_to_h5s):
        self.file_paths = glob.glob(path_to_h5s + '*/*.h5')
        self.file_index = 0

    def get_current_sample(self):
        h5file = h5py.File(self.file_paths[self.file_index], 'r')
        video = np.transpose(h5file['tissue/data'], [2, 1, 0])
        h5file.close()
        return video[0, :, :]

    def get_next_sample(self):
        if self.file_index < len(self.file_paths):
            self.file_index += 1
            return self.get_current_sample()
        else:
            return None

    def get_previous_sample(self):
        if self.file_index > 0:
            self.file_index -= 1
        return self.get_current_sample()


backend = Backend('../../data/interim/')
fig, ax = plt.subplots()
ax.imshow(backend.get_current_sample())
cross = MouseCross(ax, marker='x', markersize=10,
                   color='red',)
fig.canvas.mpl_connect('motion_notify_event', cross.show_cross)
fig.canvas.mpl_connect('button_release_event', cross.place_cross)
plt.tight_layout()
plt.show()
