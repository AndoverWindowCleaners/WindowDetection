import cv2
import matplotlib.pyplot as plt
import numpy as np

def inv_CRF(x):
    return x ** 0.8

base_dir = '../data/videos/'
# ,'20210710-195636.avi','20210710-200436.avi','20210710-195957.avi','20210710-194508.avi'
video_path = '59.mov'
video_path = base_dir + video_path
print(video_path)
capture = cv2.VideoCapture(video_path)

length = int(capture.get(7))
frames = []
for i in range(length):
    ret, frame = capture.read()
    frames.append(frame)

print(len(frames))

ax[1].imshow(cv2.cvtColor(frames[500], cv2.COLOR_BGR2RGB))
plt.show()
