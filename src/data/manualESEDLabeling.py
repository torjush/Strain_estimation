from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data-file', type=str,
                    help='Path to h5 file to label')
args = parser.parse_args()


class MainWindow(QMainWindow):

    def __init__(self, backend):
        super().__init__()

        self.backend = backend

        self.setWindowTitle("Label frames")
        self.build_UI()
        self.show()

    def build_UI(self):
        self.setGeometry(200, 200, self.backend.image_dims[1], self.backend.image_dims[0])

        self.pixmap_label = QLabel()
        self.pixmap_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.pixmap_label.resize(self.backend.image_dims[1], self.backend.image_dims[0])
        self.pixmap_label.setAlignment(Qt.AlignCenter)

        self.img_array = self.backend.get_current_image()
        self.display_image()

        self.setCentralWidget(self.pixmap_label)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Left:
            self.img_array = backend.get_previous_image()
            self.display_image()
        elif e.key() == Qt.Key_Right:
            self.img_array = backend.get_next_image()
            self.display_image()
        elif e.key() == Qt.Key_D:
            self.backend.update_label(1)
            self.img_array = backend.get_current_image()
            self.display_image()
        elif e.key() == Qt.Key_S:
            self.backend.update_label(2)
            self.img_array = backend.get_current_image()
            self.display_image()
        elif e.key() == Qt.Key_Space:
            self.backend.save_labels()

    def display_image(self):
        qimage = QImage(self.img_array, self.backend.image_dims[1],
                        self.backend.image_dims[0], self.backend.image_dims[1],
                        QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(self.backend.image_dims[1], self.backend.image_dims[0], Qt.KeepAspectRatio)

        self.pixmap_label.setPixmap(pixmap)


class Backend():
    def __init__(self, data_file):
        self.data_file = data_file
        with h5py.File(self.data_file, 'r') as f:
            ultrasound = f['tissue']
            video = np.transpose(ultrasound['data'], [2, 1, 0])

        self.images = video.astype('uint8')
        self.image_dims = video.shape[1:]
        self.image_idx = 0

        self.labels = np.zeros(self.images.shape[0])

    def get_next_image(self):
        if self.image_idx < self.images.shape[0] - 1:
            self.image_idx += 1
        return self.images[self.image_idx, :, :].copy()

    def get_previous_image(self):
        if self.image_idx > 0:
            self.image_idx -= 1
        return self.images[self.image_idx, :, :].copy()

    def get_current_image(self):
        return self.images[self.image_idx, :, :].copy()

    def update_label(self, label):
        if label == 1 or label == 2:
            if self.labels[self.image_idx] != 0:
                self.labels[self.image_idx] = 0
                self.images[self.image_idx, -5:, :] = 0
                print("Reset label to 0")
            else:
                self.labels[self.image_idx] = label
                self.images[self.image_idx, -5:, :] = 255
                print(f"Set label to {label}")

    def save_labels(self):
        with h5py.File(self.data_file, 'r+') as f:
            ultrasound = f['tissue']
            try:
                ultrasound.create_dataset('ds_labels', data=self.labels)
            except RuntimeError as e:
                if e.__str__() == "Unable to create link (name already exists)":
                    ultrasound['ds_labels'][...] = self.labels
                else:
                    raise e
            print("Saved Labels")


print(f'loading {args.data_file}')
backend = Backend(args.data_file)
app = QApplication([])
window = MainWindow(backend)

app.exec_()
