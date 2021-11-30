import cv2
import os

vid_root = os.path.join('..','data','videos')
img_root = os.path.join('..','data','images')
for name in os.listdir(vid_root):
    EVERY_OTHER = 10 # how many frames per image, starting from first frame
    file_name = vid_root + name
    new_folder = img_root + name.split('.')[0]
    print(f'generating {new_folder}')

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    cap = cv2.VideoCapture(file_name)

    i = 0
    num_images = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if i == 0:
                new_name = "%04d" % num_images
                cv2.imwrite(os.path.join(new_folder, new_name + '.jpg'), frame)
                num_images += 1
        else:
            break
        i = (i + 1) % EVERY_OTHER

    cap.release()
