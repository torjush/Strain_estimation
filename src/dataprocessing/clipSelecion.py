import h5py
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import glob
from functools import partial
import shutil


def ultraSoundAnimation(video, points=None, fps=24, with_colorbar=False):
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
    return fig, anim


PROJ_ROOT = '../../'
data_path = os.path.join(PROJ_ROOT, 'data/interim/ed_2_ed_clips')

data_files = glob.glob(os.path.join(data_path, '*/*.h5'))
sorted_path = os.path.join(PROJ_ROOT, 'data/interim/sorted_clips')


def pressOnFile(event, data_file):
    """Keypress event handler"""

    file_id = data_file.split('/')[-2:]
    rel_file_path = os.path.join(file_id[0], file_id[1])
    if event.key == 'backspace':
        new_file_path = os.path.join(sorted_path, 'dont_keep', rel_file_path)
        print('Not keeping clip')
        print(f'Moving {data_file} to {new_file_path}')
        if not os.path.exists(os.path.join(sorted_path, 'dont_keep', file_id[0])):
            os.makedirs(os.path.join(sorted_path, 'dont_keep', file_id[0]))
        shutil.move(data_file, new_file_path)

    elif event.key == 'enter':
        print('Keeping clip')
        new_file_path = os.path.join(sorted_path, 'keep', rel_file_path)
        print(f'Moving {data_file} to {new_file_path}')
        if not os.path.exists(os.path.join(sorted_path, 'keep', file_id[0])):
            os.makedirs(os.path.join(sorted_path, 'keep', file_id[0]))
        shutil.move(data_file, new_file_path)
    else:
        print(f'pressed {event.key}')
    plt.close()


# try:
#     with open('sort_log.txt', 'r') as log_file:
#         log_lines = log_file.readlines()
#     if log_lines != []:
#         for line in log_lines:
#             try:
#                 data_files.remove(line[:-1])  # without newline
#             except ValueError:
#                 print("Deleted this file")
# except FileNotFoundError:
#     print("First run, no logfile exists")

for data_file in data_files:
    print(f'Loading file: {data_file}')
    with h5py.File(data_file) as data:
        video = data['tissue/data'][:]
        fps = 1 / (data['tissue/times'][3] - data['tissue/times'][2])

    press = partial(pressOnFile, data_file=data_file)
    fig, anim = ultraSoundAnimation(video, fps=fps)
    fig.canvas.mpl_connect('key_press_event', press)
    plt.show()

    # # Write file to log as done
    # with open('sort_log.txt', 'a+') as log_file:
    #     log_file.write(data_file + '\n')
