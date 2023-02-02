import os
import cv2
import torch
import numpy as np

from PIL import Image
from skimage import transform as trans
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from collections import defaultdict


class IJB_Dataset(Dataset):
    def __init__(self, root_dir, names, lmks, size=(112, 112)):
        super(IJB_Dataset, self).__init__()
        self.root_dir = root_dir
        self.names = names
        self.lmks = lmks

        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src

        self.ccrop = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.hflip = transforms.Compose([
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.size = size

    def __getitem__(self, index):
        img = cv2.imread(self.root_dir + os.sep + self.names[index])
        tform = trans.SimilarityTransform()
        tform.estimate(self.lmks[index], self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(img, M, self.size, borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ccropped = self.ccrop(img)
        fliped = self.hflip(img)
        return ccropped, fliped

    def __len__(self):
        return len(self.names)


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform):
        super(FaceDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        classes, class_to_idx = self._find_classes(self.root_dir)
        samples, label_to_indexes = self._make_dataset(self.root_dir, class_to_idx)
        self.samples = samples 
        self.class_to_idx = class_to_idx 
        self.label_to_indexes = label_to_indexes 
        self.classes = sorted(self.label_to_indexes.keys())
        
        self.scores = np.zeros(len(self.samples), dtype=np.float32)
        self.p = 0.2

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self, root_dir, class_to_idx):
        images = []
        label2index = defaultdict(list)
        image_index = 0
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(root_dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname[-4:] not in ['.jpg', '.png']:
                        continue

                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    label2index[class_to_idx[target]].append(image_index)
                    image_index += 1

        return images, label2index

    def _get_thres(self, n=100000):
        n = min(n, len(self.scores))
        sub_scores = np.random.choice(self.scores, n)
        indexes = sub_scores.argsort()
        th1 = int(n*self.p)
        return sub_scores[indexes[[th1]]]

    def find_specfic(self, cid, lower_bound=0.0, upper_bound=1.0, n=1):
        img_ids = np.array(self.label_to_indexes[cid], dtype=np.int64)
        scores = self.scores[img_ids]

        candidates = img_ids[(scores > lower_bound) & (scores < upper_bound)]
        while len(candidates) < n:
            lower_bound -= 0.05
            upper_bound += 0.05
            candidates = img_ids[(scores > lower_bound) & (scores < upper_bound)]         
            if lower_bound < 0 and upper_bound > 1.0 and len(candidates) < n:
                candidates = candidates.repeat(1 + n // len(candidates))
                break

        indexes = np.arange(len(candidates))
        np.random.shuffle(indexes)
        return candidates[indexes[:n]]

    def get_easy_hard(self, labels, ne=7, nh=1, k=32):
        th = self._get_thres()

        # select k unique classes from given labels
        unique_labels = np.unique(labels)
        indexes = np.arange(len(unique_labels))
        np.random.shuffle(indexes)
        selected_classes = unique_labels[indexes[:k]]

        samples = []
        for cid in selected_classes:
            easy_ids = self.find_specfic(cid, th, 1.0, ne)
            hard_ids = self.find_specfic(cid, 0.0, th, nh)
            for i in np.concatenate([easy_ids, hard_ids], axis=0):
                sample, _, _ = self.__getitem__(i)
                samples.append(sample)

        samples = torch.stack(samples) 
        return samples, selected_classes

    def update_scores(self, scores, indexes, alpha=0.9):
        self.scores[indexes] = alpha*self.scores[indexes] + (1-alpha)*scores

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, index

    def __len__(self):
        return len(self.samples)

