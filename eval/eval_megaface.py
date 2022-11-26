import torch
import torch.nn.functional as F
from torch.multiprocessing import Pool
from prettytable import PrettyTable
import numpy as np
import os
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class MegaFace(Dataset):
    def __init__(self, src_dir, file_list):
        self.file_list = file_list
        self.src_dir = src_dir

    def __len__(self,):
        return len(self.file_list)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.src_dir, self.file_list[index]))
        img = cv2.resize(img, (112, 112))

        if img.ndim == 2:
            buf = np.zeros((3, img.shape[0], img.shape[1]), dtype=np.uint8)
            buf[0] = img
            buf[1] = img
            buf[2] = img
            return (buf.astype(np.float32) - 127.5) * 0.0078125
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img.transpose((2, 0, 1)) - 127.5) * 0.0078125
            return img.astype(np.float32)

def find_score(far, vr, target=1e-4):
    l = 0
    u = far.size - 1
    e = -1
    while u - l > 1:
        mid = (l + u) // 2
        if far[mid] == target:
            if target != 0:
                return vr[mid]
            else:
                e = mid
                break
        elif far[mid] < target:
            u = mid
        else:
            l = mid
    if target == 0:
        i = e
        while i >= 0:
            if far[i] != 0:
                break
            i -= 1
        if i >= 0:
            return vr[i + 1]
        else:
            return vr[u]
    if target != 0 and far[l] / target >= 8:
        return 0.0, 0
    nearest_point = (target - far[l]) / (far[u] - far[l]) * (vr[u] - vr[l]) + vr[l]
    return nearest_point

def compute_roc(G, P, G_labels, P_labels, steps=10, low=-1.0, high=1.0 + 1e-5, num_bins=10000):
    """
    perform matrix multiplication between gallery and probe and get ROC. As G and P might be so large that it cannot
    be fed into gpu memory so we should do partitioned matrix multiplication step by step. 
    I have to confess that rank1 might be higher than the result when strictly follows original protocol.
    But it may become too complicated so I choose the implementation below.
    :param G: gallery matrix, shape is #gallery x #dim
    :param P: probe matrix, shape is #probe x #dim
    :param G_labels: int32, gallery labels, shape is #gallerys x 1
    :param P_labels: int32, probe labels, shape is #probes x 1
    :param steps: how many rows we wish to select from P to do multiplication with G
    :param low: low threshold
    :param high: high threshold
    :param num_bins: num of bins
    :return: positive and negative bins
    """
    pos_bins = torch.zeros(num_bins).cuda(0)
    neg_bins = torch.zeros(num_bins).cuda(0)

    top1_sims_list = []

    indices = np.arange(P.shape[0])
    for i in tqdm(range(0, P.shape[0], steps), 'computing roc'):
        index = torch.from_numpy(indices[i:i + steps]).long()
        p = P[index] 
        pg = F.linear(p, G) # [#step, #gallery]
        binary_label = P_labels[index] == G_labels.T # [#steps, #gallery]
        sim, index = torch.topk(pg, 2, 1) # [#step, 2]
        top1_sim = sim[:, 1] # exclude itself 
        top1_index = index[:, 1].view(-1, 1) # exclude itself so we use top2
        bool_hit_flag = torch.gather(binary_label, 1, top1_index).squeeze()
        hit_candidate_sim = top1_sim[bool_hit_flag]
        if hit_candidate_sim.numel() > 0:
            top1_sims_list.extend(hit_candidate_sim.tolist())
        pos_bins += torch.histc(pg[binary_label], num_bins, low, high)
        neg_bins += torch.histc(pg[~binary_label], num_bins, low, high)

    num_probe = P.shape[0]
    pos_bins[-1] -= num_probe

    # perform cumulative sum in reverse order
    num_positives = torch.sum(pos_bins).item()
    num_negatives = torch.sum(neg_bins).item()
    # logger.info('#negative samples %d, #positive samples %d' % (num_negatives, num_positives))
    far = (torch.flip(torch.cumsum(torch.flip(neg_bins.view(1, num_bins), [1]), 1), [1]).squeeze() / num_negatives).cpu().numpy()
    vr = (torch.flip(torch.cumsum(torch.flip(pos_bins.view(1, num_bins), [1]), 1), [1]).squeeze() / num_positives).cpu().numpy()
    # logger.info('find score...')

    thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    tprs = [find_score(far, vr, th) for th in thresholds]

    table = PrettyTable(['far=1e-6', 'far=1e-7', 'far=1e-8', 'far=1e-9'])
    table.add_row([round(tprs[5] * 100, 2), round(tprs[6] * 100, 2), round(tprs[7] * 100, 2), round(tprs[8] * 100, 2)])
    return tprs, table, pos_bins, neg_bins

def compute_rank_1(pfeat, dfeat, dir2range):
    tp = 0
    total = 0
    gt = dfeat.shape[0] - 1
    steps = 10
    for (d, se) in tqdm(dir2range.items(), desc='computing rank 1'):
        P = pfeat[se[0]: se[1]]
        M = P.shape[0]
        inter_scores = [] # max inter sims
        for i in range(0, M, steps):
            p = P[i:i + steps]
            m = p.shape[0]
            sims = F.linear(p, dfeat).view(m, -1).max(axis=1)[0].tolist() # [p, 512], [d, 512] -> [p, d] -> [p]
            inter_scores.extend(sims)
        intra_sims = F.linear(P, P).cpu().numpy() # [p, 512], [p, 512] -> [p, p]
        total += (M - 1) * M
        for i in range(M):
            tp += np.sum(intra_sims[i] > inter_scores[i]) - 1

    acc = tp / (total * 1.0)
    return acc


def extract_features(model, test_list, test_labels, feat_dim=512, plus_one=False, use_flip=True):
    model.eval()
    batch_size = 128

    if plus_one:
        test_feat = torch.zeros((len(test_list) + 1, feat_dim), dtype=torch.float32).cuda(0)
    else:
        test_feat = torch.zeros((len(test_list), feat_dim), dtype=torch.float32).cuda(0)

    db = MegaFace('', test_list)
    db_loader = DataLoader(db, batch_size, num_workers=4, pin_memory=False, drop_last=False)

    gid_end = 0

    with torch.no_grad():
        bid = 0
        for i, img in enumerate(tqdm(db_loader)):
            img = img.cuda(0)
            feat = model(img)
            if use_flip:
                fliped_img = torch.flip(img, dims=[3]).cuda(0)
                feat += model(fliped_img)
            gid_start = gid_end
            gid_end += img.shape[0]
            test_feat[gid_start:gid_end] = feat

    if plus_one:
        test_label = torch.from_numpy(np.array(test_labels + [-1])).cuda(0)
    else:
        test_label = torch.from_numpy(np.array(test_labels)).cuda(0)
    test_feat = F.normalize(test_feat, dim=-1)
    return test_feat, test_label


def prepare_probe(root, noise_list=None, start_label=0):
    noise_count = 0
    if noise_list is not None:
        with open(noise_list, 'r') as f:
            noise_list = [line.strip() for line in f.readlines() if not line.startswith('#')]

    dir2labels = {}
    dir2range = {}
    gp_list = []
    gp_label = []
    pd = os.listdir(root)
    next_label = start_label
    num_images = 0
    start_index = 0
    end_index = 0
    images_per_id = 0
    for d in pd:
        sd = os.path.join(root, d)
        if os.path.isdir(sd):
            files = os.listdir(sd)
            has_image = False
            for e in files:
                if noise_list is not None and e in noise_list:
                    noise_count += 1
                    continue
                ext = os.path.splitext(e)[1].lower()
                if ext in ('.jpg', '.png'):
                    gp_label.append(next_label)
                    gp_list.append(os.path.join(sd, e))
                    has_image = True
                    num_images += 1
                    images_per_id += 1
            if has_image:
                start_index = end_index
                end_index += images_per_id
                dir2range[d] = [start_index, end_index]
                dir2labels[d] = next_label
                images_per_id = 0
                next_label += 1
    print('probe noise count:', noise_count)
    return gp_list, gp_label, dir2labels, dir2range


def prepare_distractor(root, noise_list=None):
    noise_count = 0
    if noise_list is not None:
        with open(noise_list, 'r') as f:
            noise_list = [line.strip() for line in f.readlines() if not line.startswith('#')]

    distractor_label = []
    distractor_list = []
    for iter_root, sub_dirs, files in os.walk(root):
        for e in files:
            ppd = os.path.basename(os.path.dirname(iter_root))
            pd = os.path.basename(iter_root)
            if noise_list is not None and f'{ppd}/{pd}/{e}' in noise_list:
                noise_count += 1
                continue
            ext = os.path.splitext(e)[1].lower()
            if ext in ('.jpg', '.png'):
                distractor_label.append(-1)
                distractor_list.append(os.path.join(iter_root, e))
    print('distractor noise count:', noise_count)
    return distractor_list, distractor_label


def preform_eval(model, save_root, root='/datasets/face/megaface_testpack/'):
    probe_dir = f'{root}/facescrub_images'
    distractor_dir = f'{root}/megaface_images'

    probe_list, probe_label, dir2labels, dir2range = prepare_probe(probe_dir, noise_list=f'{root}/facescrub_noises.txt')
    distractor_list, distractor_label = prepare_distractor(distractor_dir, noise_list=f'{root}/megaface_noises.txt')

    pfeat, plabel = extract_features(model, probe_list, probe_label)
    dfeat, dlabel = extract_features(model, distractor_list, distractor_label)

    # identification
    acc = compute_rank_1(pfeat, dfeat, dir2range)

    # verification
    gfeat = torch.cat([pfeat, dfeat])
    glabel = torch.cat([plabel, dlabel])

    glabel = glabel.view(-1, 1)
    plabel = plabel.view(-1, 1)

    tprs, table, _, _ = compute_roc(gfeat, pfeat, glabel, plabel)
    return acc, tprs, table



