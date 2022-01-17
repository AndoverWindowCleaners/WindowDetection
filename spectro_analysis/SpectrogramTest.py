from operator import inv
from matplotlib import mlab
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def inv_CRF(x):
    return x**0.8

base_dir = '../data/videos/'
#,'20210710-195636.avi','20210710-200436.avi','20210710-195957.avi','20210710-194508.avi'
video_paths = [
    '46.mov',
    '47.mov',
    '48.mov',
    '49.mov'
]
video_paths = [base_dir+video_path for video_path in video_paths]
captures = [cv2.VideoCapture(video_path) for video_path in video_paths]
lengths = [int(capture.get(7)) for capture in captures]
fpss = [capture.get(5) for capture in captures]
durations = [length/fps for length,fps in zip(lengths,fpss)]
print(lengths)
all_frames = [np.zeros((length,9,16,3),dtype=np.float16) for length in lengths] # resizes W:360, H:640, 9x16

def image_pooling(image, new_width, new_height):
    return cv2.resize(image, (new_height, new_width), interpolation=cv2.INTER_AREA)

for j,length,capture in zip(range(len(lengths)),lengths,captures):
    for i in range(length):
        ret, frame = capture.read()
        if ret:
            frame = image_pooling(frame, 9, 16)
            all_frames[j][i] = frame

all_frames = [inv_CRF(all_frame[3:100]/255.0) for all_frame in all_frames]
print(np.mean(all_frames[0]))
all_spectrs = []
all_freqs = []
all_times = []
for all_frame in all_frames:
    freqs, times, spectr = signal.spectrogram(all_frame, fs=30.0, window=('hamming'), noverlap=13,  nperseg=14, axis=0, mode='magnitude') 
    # all_frame has shape (T,12,9,3) (T)--how many frames are in the video
    # output (N, 12, 9, 3, T') -- applies fourier analysis to each all_shape[start:end,i,j,k] --> (N)
    # output (N, 12, 9, 3, T')


    # spectr = spectr[3:]/np.sum(spectr[0:3],axis=0)
    # freqs = freqs[3:]
    print(freqs[:10])
    all_times.append(times)
    all_freqs.append(freqs)
    all_spectrs.append(spectr)

fig, axes = plt.subplots(2,2)
for i,all_frame,ax in zip(range(len(all_frames)),all_frames,axes.flatten()):
    freqs,times,spectr = all_freqs[i],all_times[i],all_spectrs[i]
    print(freqs.shape, spectr.shape)
    ax.pcolormesh(times, freqs, np.mean(spectr[:,3:4,4:5,1:2,:],axis=(1,2,3)), shading='nearest')
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html
    
plt.show()


