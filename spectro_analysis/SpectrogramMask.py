import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from matplotlib.widgets import Slider, Button

freq_i = 3
time_i = 200

base_dir = '../data/videos/'
video_num = '67'
video = f'{video_num}.mov'


def plot_spect_mask(scale):
    global freq_i, time_i, base_dir, video, video_num

    height = 16*scale
    width = 9*scale

    def inv_CRF(x):
        return x ** 0.8

    video_path = base_dir + video
    capture = cv2.VideoCapture(video_path)
    length = int(capture.get(7))
    fps = capture.get(5)
    duration = length / fps

    # Scaling:
    all_frame = np.zeros((length, height, width, 3), dtype=np.float32)  # resizes 360 x 640 to 12 x 9


    # More Scaling:
    def image_pooling(image, new_width, new_height):
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


    # Even More Scaling:

    for i in range(length):
        ret, frame = capture.read()
        # (640, 360, 3)
        if ret:
            frame = image_pooling(frame, width, height)
            all_frame[i] = frame
    all_frame /= 255.0
    all_frame_processed = inv_CRF(all_frame)

    # Output: Determines the value for each column
    freqs, times, spectr = signal.spectrogram(all_frame_processed, fs=30.0, window=('hamming'), noverlap=13, nperseg=14, axis=0,
                                              mode='magnitude')
    # freqs is all the frequency values used in the fourier analysis
    # times is the time frames for which we generated a fourier decomposition

    # all_frame_processed has shape (T,12,9,3) (T)--how many frames are in the video
    # output (N, 12, 9, 3, T') -- applies fourier analysis to each all_shape[start:end,i,j,k] --> (N)
    # output (N, 12, 9, 3, T')
    spectr = np.mean(spectr, axis=3)

    spectr = spectr[freq_i, :, :, time_i]

    img_i = int(times[time_i]*30)

    fig, ax = plt.subplots(1, 2)

    ax[0].pcolormesh(list(range(spectr.shape[1])), list(range(spectr.shape[0])), spectr, shading='nearest')

    img_frame = all_frame[img_i]


    ax[1].imshow(cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB))
    ax[0].invert_yaxis()

    plt.savefig(os.path.join("spectro_masks",f"spec{img_i}_{freq_i}_{height}.png"))
    capture.release()

for scale in range(1, 11):
    plot_spect_mask(scale)

