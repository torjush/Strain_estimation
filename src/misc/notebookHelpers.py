import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
plt.rcParams['animation.html'] = 'html5'


def ultraSoundAnimation(video, points=None, fps=24, with_colorbar=False):
    """
    Play ultrasound recording in a jupyter notebook

    Args:
      video(numpy array): Ultrasound recording as 3d array
                          (frame, height, width)
      points(numpy array): 3D array containing mitral annulus points
                           (frame, [left right], [x y])
      fps(int): Frame rate of ultrasound recording
      with_colorbar(bool): Wether or not to show a colorbar
                           beside the animation

    Returns:
      anim(matplotlib.animation.FuncAnimation): Animation to be
                                                played in a notebook
    """

    fig, ax = plt.subplots()
    line = ax.imshow(video[0, :, :], cmap='Greys_r')
    if points is not None:
        line2 = ax.scatter(points[0, 0, 0], points[0, 0, 0], color='red')
        line3 = ax.scatter(points[0, 1, 0], points[0, 1, 1], color='red')
    if with_colorbar:
        ax.figure.colorbar(line, ax=ax)

    def init():
        line.set_data(video[0, :, :])
        if points is not None:
            line2.set_offsets([points[0, 0, 0], points[0, 0, 1]])
            line3.set_offsets([points[0, 1, 0], points[0, 1, 1]])
            return (line, line2, line3)
        else:
            return (line, )

    def animate(i):
        line.set_data(video[i, :, :])
        if points is not None:
            line2.set_offsets([points[i, 0, 0], points[i, 0, 1]])
            line3.set_offsets([points[i, 1, 0], points[i, 1, 1]])
            return (line, line2, line3)
        else:
            return (line, )

    interval = 1 / fps * 1000  # from framerate to ms
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=video.shape[0], interval=interval,
                                   blit=True)
    plt.close()
    return anim


def video_and_fps_from_h5py(data_file, video_type='b-mode'):
    """
    Extract video as array and fps from hdf5 dataset

    Args:
      data_file(h5py File): hdf5 file containing ultrasound data and timestamps
      video_type(string): Wether to extract b-mode or tvi data as video
                          Takes "b-mode" or "tvi"

    Raises:
      RuntimeError: video_type must be either "b-mode" or "tvi"

    Returns:
      video(numpy array): Ultrasound recording as a
                          3d array(frame, height, width)
      fps(int): Frame rate of ultrasound recording
    """

    video_type = video_type.lower()
    if video_type == 'b-mode':
        dataset = data_file['tissue']
    elif video_type == 'tvi':
        dataset = data_file['TVI']
    else:
        raise RuntimeError("video_type must be either 'b-mode' or 'tvi'")

    video = dataset['data']
    video = np.transpose(video, [2, 1, 0])

    dt = dataset['times'][1] - dataset['times'][0]
    fps = int(1 / dt)

    return video, fps
