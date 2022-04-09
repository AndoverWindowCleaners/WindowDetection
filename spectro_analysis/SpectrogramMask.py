from operator import inv
from matplotlib import mlab
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def inv_CRF(x):
    return x ** 0.8


base_dir = '../data/videos/'
# ,'20210710-195636.avi','20210710-200436.avi','20210710-195957.avi','20210710-194508.avi'
video_path = 'Folder 59 W.mov'
video_path = base_dir + video_path
print(video_path)
capture = cv2.VideoCapture(video_path)
length = int(capture.get(7))
fps = capture.get(5)
duration = length / fps
print(length)

# Scaling:
all_frame = np.zeros((length, 48, 27, 3), dtype=np.float16)  # resizes 360 x 640 to 12 x 9


# More Scaling:
def image_pooling(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


# Even More Scaling:

for i in range(length):
    ret, frame = capture.read()
    # (640, 360, 3)
    if ret:
        frame = image_pooling(frame, 27, 48)
        all_frame[i] = frame

all_frame = inv_CRF(all_frame / 255.0)
print(np.mean(all_frame))

# Output: Determines the value for each column
freqs, times, spectr = signal.spectrogram(all_frame, fs=30.0, window=('hamming'), noverlap=13, nperseg=14, axis=0,
                                          mode='magnitude')
# freqs is all the frequency values used in the fourier analysis
# times is the time frames for which we generated a fourier decomposition
print(freqs.shape)
print(times.shape)
print(spectr.shape)
# all_frame has shape (T,12,9,3) (T)--how many frames are in the video
# output (N, 12, 9, 3, T') -- applies fourier analysis to each all_shape[start:end,i,j,k] --> (N)
# output (N, 12, 9, 3, T')
spectr = np.mean(spectr, axis=3)

print(freqs)

spectr = spectr[3, :, :, 9]

fig, ax = plt.subplots(1, 1)

ax.pcolormesh(list(range(spectr.shape[1])), list(range(spectr.shape[0])), spectr, shading='nearest')

ax.invert_yaxis()

plt.show()