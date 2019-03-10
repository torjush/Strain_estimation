import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import skimage

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir, os.pardir))
data_path = os.path.join(PROJ_ROOT, 'data/interim/sorted_clips/keep')
processed_data_path = os.path.join(PROJ_ROOT, 'data/interim/strain_point')

h5files = glob.glob(os.path.join(data_path, '*/*.h5'))


def cm2Pixel(num_cm, data_file, dim):
    pixel_size_cm = data_file['tissue/pixelsize'][dim] * 100
    return int(round(num_cm / pixel_size_cm))


for h5file in h5files:
    with h5py.File(h5file) as data:
        video = data['tissue/data'][:]
        frame = video[0, :, :].copy()
        mitral_points = data['tissue/ma_points'][0, :, :].astype('int')

        times = data['tissue/times'][:]
        ds_labels = data['tissue/ds_labels'][:]

        ecg = data['ecg/ecg_data'][:]
        ecg_times = data['ecg/ecg_times'][:]

        # Check for non-visible points(marked outside valid image area)
        left_point_val = frame[mitral_points[0, 1], mitral_points[0, 0]]
        left_point_valid = left_point_val != 0

        right_point_val = frame[mitral_points[1, 1], mitral_points[1, 0]]
        right_point_valid = right_point_val != 0

        if left_point_valid:
            search_box_left = np.array(
                [[mitral_points[0, 0] - cm2Pixel(3, data, 0),
                  mitral_points[0, 0] + cm2Pixel(0, data, 0)],
                 [mitral_points[0, 1] + cm2Pixel(2., data, 1),
                  mitral_points[0, 1] + cm2Pixel(4., data, 1)]]).astype('int')

        if right_point_valid:
            search_box_right = np.array(
                [[mitral_points[1, 0] - cm2Pixel(0, data, 0),
                  mitral_points[1, 0] + cm2Pixel(3, data, 0)],
                 [mitral_points[1, 1] + cm2Pixel(1.9, data, 1),
                  mitral_points[1, 1] + cm2Pixel(3.5, data, 1)]]).astype('int')
        print(h5file.split('/')[-2:])

    # Normalize frame
    frame /= 255.
    # Do filtering (sobel should only be done in the ultrasound region)
    mask = frame != 0
    kernel = np.ones((5, 5)) / 9
    mean_frame = skimage.filters.rank.mean(frame, kernel, mask=mask).astype('float')
    mean_frame /= 255.
    mean_frame /= np.max(mean_frame)
    sobel_mean_frame = skimage.filters.sobel(mean_frame, mask=mask)
    sobel_mean_frame /= np.max(sobel_mean_frame)

    # Find points
    if left_point_valid:
        weights = np.linspace(0.7, 1, search_box_left[0, 1] - search_box_left[0, 0])
        weights = np.stack([weights] * (search_box_left[1, 1] - search_box_left[1, 0]))
    
        left_sobel_search_region = sobel_mean_frame[search_box_left[1, 0]:search_box_left[1, 1],
                                                    search_box_left[0, 0]:search_box_left[0, 1]]
        left_mean_search_region = mean_frame[search_box_left[1, 0]:search_box_left[1, 1],
                                             search_box_left[0, 0]:search_box_left[0, 1]]

        left_search_region = left_sobel_search_region +\
            left_mean_search_region + weights

        flattened_index = np.argmax(left_search_region)
        strain_point_left = np.unravel_index(flattened_index, left_search_region.shape)
        strain_point_left = np.array([strain_point_left[1] + search_box_left[0, 0],
                                      strain_point_left[0] + search_box_left[1, 0]])

    if right_point_valid:
        weights = np.linspace(1, 0.7, search_box_right[0, 1] - search_box_right[0, 0])
        weights = np.stack([weights] * (search_box_right[1, 1] - search_box_right[1, 0]))

        right_sobel_search_region = sobel_mean_frame[search_box_right[1, 0]:search_box_right[1, 1],
                                                     search_box_right[0, 0]:search_box_right[0, 1]]
        right_mean_search_region = mean_frame[search_box_right[1, 0]:search_box_right[1, 1],
                                              search_box_right[0, 0]:search_box_right[0, 1]]

        right_search_region = right_sobel_search_region +\
            right_mean_search_region + weights

        flattened_index = np.argmax(right_search_region)
        strain_point_right = np.unravel_index(flattened_index, right_search_region.shape)
        strain_point_right = np.array([strain_point_right[1] + search_box_right[0, 0],
                                       strain_point_right[0] + search_box_right[1, 0]])

    def savePoints():
        file_id = h5file.split('/')[-2:]
        if not os.path.exists(os.path.join(processed_data_path, file_id[0])):
            os.makedirs(os.path.join(processed_data_path, file_id[0]))
        file_id = os.path.join(file_id[0], file_id[1])

        with h5py.File(os.path.join(processed_data_path, file_id), 'w') as new_data:
            new_data.create_dataset('tissue/data', data=video)
            new_data.create_dataset('tissue/times', data=times)
            new_data.create_dataset('tissue/ds_labels', data=ds_labels)
            if left_point_valid:
                left_points = np.vstack((mitral_points[0, :], strain_point_left))
            else:
                left_points = np.empty(0)
            new_data.create_dataset('tissue/left_points', data=left_points)

            if right_point_valid:
                right_points = np.vstack((mitral_points[1, :], strain_point_right))
            else:
                right_points = np.empty(0)
            new_data.create_dataset('tissue/right_points', data=right_points)

            new_data.create_dataset('ecg/ecg_data', data=ecg)
            new_data.create_dataset('ecg/ecg_times', data=ecg_times)

    def press(event):
        if event.key == 'enter':
            plt.close()
            savePoints()
        elif event.key == 'backspace':
            plt.close()
        elif event.key == 'q':
            plt.close()
            quit()

    fig, ax = plt.subplots(ncols=3, figsize=(16, 10))
    ax[0].imshow(frame, cmap='Greys_r')
    ax[0].scatter(mitral_points[0, 0], mitral_points[0, 1], color='red')
    ax[0].scatter(mitral_points[1, 0], mitral_points[1, 1], color='red')

    if left_point_valid:
        ax[0].scatter(strain_point_left[0], strain_point_left[1], color='red')

        rectangle = plt.Rectangle((search_box_left[0, 0], search_box_left[1, 0]),
                                  search_box_left[0, 1] - search_box_left[0, 0],
                                  search_box_left[1, 1] - search_box_left[1, 0],
                                  color='r', fill=False)
        ax[0].add_patch(rectangle)

    if right_point_valid:
        ax[0].scatter(strain_point_right[0], strain_point_right[1], color='red')
        rectangle = plt.Rectangle((search_box_right[0, 0], search_box_right[1, 0]),
                                  search_box_right[0, 1] - search_box_right[0, 0],
                                  search_box_right[1, 1] - search_box_right[1, 0],
                                  color='r', fill=False)
        ax[0].add_patch(rectangle)
    ax[0].set_title('Raw image')
    ax[0].axis('off')

    ax[1].imshow(mean_frame, cmap='Greys_r')
    ax[1].scatter(mitral_points[0, 0], mitral_points[0, 1], color='red')
    ax[1].scatter(mitral_points[1, 0], mitral_points[1, 1], color='red')

    if left_point_valid:
        ax[1].scatter(strain_point_left[0], strain_point_left[1], color='red')

        rectangle = plt.Rectangle((search_box_left[0, 0], search_box_left[1, 0]),
                                  search_box_left[0, 1] - search_box_left[0, 0],
                                  search_box_left[1, 1] - search_box_left[1, 0],
                                  color='r', fill=False)
        ax[1].add_patch(rectangle)

    if right_point_valid:
        ax[1].scatter(strain_point_right[0], strain_point_right[1], color='red')
        rectangle = plt.Rectangle((search_box_right[0, 0], search_box_right[1, 0]),
                                  search_box_right[0, 1] - search_box_right[0, 0],
                                  search_box_right[1, 1] - search_box_right[1, 0],
                                  color='r', fill=False)
        ax[1].add_patch(rectangle)
    ax[1].set_title('5x5 mean filtered image')
    ax[1].axis('off')

    ax[2].imshow(sobel_mean_frame, cmap='Greys_r')
    ax[2].scatter(mitral_points[0, 0], mitral_points[0, 1], color='red')
    ax[2].scatter(mitral_points[1, 0], mitral_points[1, 1], color='red')

    if left_point_valid:
        ax[2].scatter(strain_point_left[0], strain_point_left[1], color='red')

        rectangle = plt.Rectangle((search_box_left[0, 0], search_box_left[1, 0]),
                                  search_box_left[0, 1] - search_box_left[0, 0],
                                  search_box_left[1, 1] - search_box_left[1, 0],
                                  color='r', fill=False)
        ax[2].add_patch(rectangle)

    if right_point_valid:
        ax[2].scatter(strain_point_right[0], strain_point_right[1], color='red')
        rectangle = plt.Rectangle((search_box_right[0, 0], search_box_right[1, 0]),
                                  search_box_right[0, 1] - search_box_right[0, 0],
                                  search_box_right[1, 1] - search_box_right[1, 0],
                                  color='r', fill=False)
        ax[2].add_patch(rectangle)
    ax[2].set_title('5x5 mean + sobel filtered image')
    ax[2].axis('off')

    fig.canvas.mpl_connect('key_press_event', press)
    plt.tight_layout()
    plt.show()
