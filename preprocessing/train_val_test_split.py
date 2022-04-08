import sys
sys.path.append('../')
import os
import json
import random
import copy

anno_root = os.path.join('..','data','annotations')
anno_name = 'annotations_27.json'
anno_path = os.path.join(anno_root, anno_name)

video_names = [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]
video_count = len(video_names)

train_count = int(video_count*0.6)
val_count = int(video_count*0.2)
test_count = video_count - train_count - val_count

with open(anno_path, 'r') as f:
	data = json.load(f)


train_video_names = random.sample(video_names, train_count)
video_names = [name for name in video_names if name not in train_video_names]
val_video_names = random.sample(video_names, val_count)
test_video_names = [name for name in video_names if name not in val_video_names]

print(train_video_names)
print(val_video_names)
print(test_video_names)

dataset0 = {'images':[], 'spectrograms': [], 'categories': data['categories'], 'annotations': []}

def create_dataset(videos, dataset):
	chosen_id = set([])
	for image in data['images']:
		img_vid_name = int(image['file_name'].split('/')[0])
		if img_vid_name not in videos : continue
		dataset['images'].append(copy.deepcopy(image))
		chosen_id.add(image['id'])

	for spect in data['spectrograms']:
		spect_vid_name = int(spect['file_name'].split('/')[0])
		if spect_vid_name not in videos : continue
		dataset['spectrograms'].append(copy.deepcopy(spect))

	for anno in data['annotations']:
		if anno['image_id'] not in chosen_id : continue
		dataset['annotations'].append(copy.deepcopy(anno))
	return dataset

train_dataset = create_dataset(train_video_names, copy.deepcopy(dataset0))
train_name = 'annotations_27_train.json'
train_path = os.path.join(anno_root, train_name)
with open(train_path, 'w') as f:
	json.dump(train_dataset, f)

val_dataset = create_dataset(val_video_names, copy.deepcopy(dataset0))
val_name = 'annotations_27_val.json'
val_path = os.path.join(anno_root, val_name)
with open(val_path, 'w') as f:
	json.dump(val_dataset, f)

test_dataset = create_dataset(test_video_names, copy.deepcopy(dataset0))
test_name = 'annotations_27_test.json'
test_path = os.path.join(anno_root, test_name)
with open(test_path, 'w') as f:
	json.dump(test_dataset, f)