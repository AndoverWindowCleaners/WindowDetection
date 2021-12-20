import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np

class WindowDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        img_root (string): Root directory where images are downloaded to.
        spec_root (string): Root directory where spectrograms are downloaded to
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, img_root, spec_root, annFile, transform=None, spec_transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.img_root = img_root
        self.spec_root = spec_root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.spec_transform = spec_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.img_root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        spectr = np.load(os.path.join(self.spec_root, path[:path.rfind('.')] + '.npy'))

        if self.spec_transform is not None:
            spectr = self.spec_transform(spectr)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, spectr, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Image Root Location: {}\n'.format(self.img_root)
        fmt_str += '    Spectrogram Root Location: {}\n'.format(self.spec_root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    