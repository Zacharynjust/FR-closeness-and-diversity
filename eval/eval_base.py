import cv2
import torch
import pickle
import numpy as np

from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from tqdm import tqdm

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds)) #  positive pairs(which is correctly predicted) / total positive pairs
    fprs = np.zeros((nrof_folds, nrof_thresholds)) #  negative pairs(which is incorrectly predicted as positive) / total positve pairs
    
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1) # distance
    bad_case = np.array([], dtype=np.int32)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

        best_threshold = thresholds[best_threshold_index]
        (hard_positive,) = np.where(actual_issame & ((best_threshold + 0.2 > dist) & (dist > best_threshold - 0.3)))
        (hard_negative,) = np.where(~actual_issame & ((best_threshold - 0.2 < dist) & (dist < best_threshold + 0.3)))
        bad_case = np.concatenate([bad_case, hard_positive, hard_negative])
        bad_case = np.unique(bad_case)

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)

    return tpr, fpr, accuracy, bad_case


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


class CarrayWapper(Dataset):
    def __init__(self, carray):
        self.carray = carray    
        
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),])

        self.flip_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.functional.hflip,
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),])

    def __getitem__(self, index):
        img = self.carray[index]
        raw_img = self.transform(img)
        flip_img = self.flip_transform(img)
        return raw_img, flip_img

    def __len__(self):
        return len(self.carray)


@torch.no_grad()
def perform_eval(backbone, carray, issame, batch_size=128, nrof_folds=10):
    idx = 0
    embeddings = []
    wapper = CarrayWapper(carray)
    dataloader = DataLoader(wapper, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    for imgs, flip_imgs in tqdm(dataloader):
        emb_batch = backbone(imgs.cuda(0)).cpu() + backbone(flip_imgs.cuda(0)).cpu()
        embeddings.append(emb_batch)

    embeddings = torch.cat(embeddings, dim=0)
    xnorm = embeddings.norm(p=2, dim=-1).mean()
    embeddings = F.normalize(embeddings, p=2, dim=-1)

    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2].numpy()
    embeddings2 = embeddings[1::2].numpy()
    tpr, fpr, accuracy, bad_case = calculate_roc(thresholds, embeddings1, embeddings2, issame)
    
    return accuracy.mean(), accuracy.std(), xnorm, bad_case

def load_bin(path, image_size=[112,112]):
    bins, issame = pickle.load(open(path, 'rb'), encoding='bytes')
    issame = np.array(issame, dtype=bool)
    imgs = np.zeros((len(issame)*2, image_size[0], image_size[1], 3), dtype=np.uint8)

    for i in range(len(issame)*2):
        _bin = np.array(bytearray(bins[i]), dtype=np.uint8)
        img = cv2.imdecode(_bin, cv2.IMREAD_UNCHANGED) # [h, w, c]
        if img.shape[1] != image_size[0]:
            img = cv2.resize(img, (image_size[1], image_size[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs[i, :] = img
    
    return imgs, issame