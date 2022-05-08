import cv2

def vid_read(file_name):
	cap = cv2.VideoCapture(file_name)
	
	frames = []
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break
		frames.append(frame)
	cap.release()
	return frames
