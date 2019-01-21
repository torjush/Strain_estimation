import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
plt.rcParams['animation.html'] = 'html5'


def ultraSoundAnimation(video, fps=24):
    """
    Play ultrasound recording in a jupyter notebook

    Args:
      video(numpy array): Ultrasound recording as 3d array
                          (frame, height, width)
      fps(int): Frame rate of ultrasound recording

    Returns:
      anim(matplotlib.animation.FuncAnimation): Animation to be
                                                played in a notebook
    """
    fig, ax = plt.subplots()
    line = ax.imshow(video[0, :, :])

    def init():
        line.set_data(video[0, :, :])
        return (line, )

    def animate(i):
        line.set_data(video[i, :, :])
        return (line, )

    interval = 1 / fps * 1000  # from framerate to ms
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=video.shape[0], interval=interval,
                                   blit=True)
    plt.close()
    return anim


def array_and_fps_from_h5py(data_file):
    """
    Extract video as array and fps from hdf5 dataset

    Args:
      data_file(h5py File): hdf5 file containing ultrasound data and timestamps

    Returns:
      film(numpy array): Ultrasound recording as 3d array(frame, height, width)
      fps(int): Frame rate of ultrasound recording
    """
    film = data_file['tissue/data']
    film = np.transpose(film, [2, 1, 0])

    dt = data_file['tissue/times'][1] - data_file['tissue/times'][0]
    fps = int(1 / dt)

    return film, fps
