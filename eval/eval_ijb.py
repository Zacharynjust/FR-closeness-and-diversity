import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))

from eval.insightface_ijb_helper.dataloader import prepare_dataloader
from eval.insightface_ijb_helper import eval_helper as eval_helper_verification

import warnings
warnings.filterwarnings("ignore")
import torch
from tqdm import tqdm


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm


def fuse_features_with_norm(stacked_embeddings, stacked_norms, fusion_method='norm_weighted_avg'):

    assert stacked_embeddings.ndim == 3 # (n_features_to_fuse, batch_size, channel)
    if stacked_norms is not None:
        assert stacked_norms.ndim == 3 # (n_features_to_fuse, batch_size, 1)
    else:
        assert fusion_method not in ['norm_weighted_avg', 'pre_norm_vector_add']

    if fusion_method == 'norm_weighted_avg':
        weights = stacked_norms / stacked_norms.sum(dim=0, keepdim=True)
        fused = (stacked_embeddings * weights).sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'pre_norm_vector_add':
        pre_norm_embeddings = stacked_embeddings * stacked_norms
        fused = pre_norm_embeddings.sum(dim=0)
        fused, fused_norm = l2_norm(fused, axis=1)
    elif fusion_method == 'average':
        fused = stacked_embeddings.sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'concat':
        fused = torch.cat([stacked_embeddings[0], stacked_embeddings[1]], dim=-1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)
    else:
        raise ValueError('not a correct fusion method', fusion_method)

    return fused, fused_norm

def infer_images(model, img_root, landmark_list_path, batch_size, use_flip_test=False, fusion_method='pre_norm_vector_add', gpu_id=0):
    img_list = open(landmark_list_path)

    files = img_list.readlines()
    faceness_scores = []
    img_paths = []
    landmarks = []
    for img_index, each_line in enumerate(files):
        name_lmk_score = each_line.strip().split(' ')
        img_path = os.path.join(img_root, name_lmk_score[0])
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        img_paths.append(img_path)
        landmarks.append(lmk)
        faceness_scores.append(name_lmk_score[-1])

    dataloader = prepare_dataloader(img_paths, landmarks, batch_size, num_workers=4, image_size=(112,112))

    model.eval()
    features = []
    norms = []
    with torch.no_grad():
        for images, idx in tqdm(dataloader):

            feature = model(images.to("cuda:{}".format(gpu_id)))
            if isinstance(feature, tuple):
                feature, norm = feature
            else:
                feature, norm = l2_norm(feature)

            if use_flip_test:
                fliped_images = torch.flip(images, dims=[3])
                flipped_feature = model(fliped_images.to("cuda:{}".format(gpu_id)))
                if isinstance(flipped_feature, tuple):
                    flipped_feature, flipped_norm = flipped_feature
                else:
                    flipped_feature, flipped_norm = l2_norm(flipped_feature)

                stacked_embeddings = torch.stack([feature, flipped_feature], dim=0)
                if norm is not None:
                    stacked_norms = torch.stack([norm, flipped_norm], dim=0)
                else:
                    stacked_norms = None

                fused_feature, fused_norm = fuse_features_with_norm(stacked_embeddings, stacked_norms, fusion_method=fusion_method)
                features.append(fused_feature.cpu().numpy())
                norms.append(fused_norm.cpu().numpy())
            else:
                features.append(feature.cpu().numpy())
                norms.append(norm.cpu().numpy())

    features = np.concatenate(features, axis=0)
    img_feats = np.array(features).astype(np.float32)
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    norms = np.concatenate(norms, axis=0)

    assert len(features) == len(img_paths)

    return img_feats, faceness_scores, norms

    
def verification(data_root, dataset_name, img_input_feats, save_path):
    templates, medias = eval_helper_verification.read_template_media_list(
        os.path.join(data_root, '%s/meta' % dataset_name, '%s_face_tid_mid.txt' % dataset_name.lower()))
    p1, p2, label = eval_helper_verification.read_template_pair_list(
        os.path.join(data_root, '%s/meta' % dataset_name,
                    '%s_template_pair_label.txt' % dataset_name.lower()))

    template_norm_feats, unique_templates = eval_helper_verification.image2template_feature(img_input_feats, templates, medias)
    score = eval_helper_verification.verification(template_norm_feats, unique_templates, p1, p2)

    # # Step 5: Get ROC Curves and TPR@FPR Table
    score_save_file = os.path.join(save_path, "verification_score.npy")
    np.save(score_save_file, score)
    result_files = [score_save_file]
    table, ret = eval_helper_verification.write_result(result_files, save_path, dataset_name, label)
    os.remove(score_save_file)
    return table, ret


def perform_eval(model, name, save_root, root='/datasets/face/IJB_release/'):
    assert name.lower() in ['ijbb', 'ijbc']
    img_root = os.path.join(root, './%s/loose_crop' % name.upper())
    landmark_list_path = os.path.join(root, './%s/meta/%s_name_5pts_score.txt' % (name.upper(), name.lower()))
    img_input_feats, faceness_scores, norms = infer_images(model=model,
                                                           img_root=img_root,
                                                           landmark_list_path=landmark_list_path,
                                                           batch_size=128)

    img_input_feats = img_input_feats * norms

    table, ret = verification(root, name.upper(), img_input_feats, save_root)
    return table, ret




