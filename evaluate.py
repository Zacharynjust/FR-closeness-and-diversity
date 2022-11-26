import os
import logging
import argparse
import torch
import numpy as np

from eval import eval_base, eval_ijb, eval_megaface
from backbone.model_irse import *
from eval_config import configurations
from utils.utils_logging import save_config, init_logging

def evaluate_base(model, name, LOG_ROOT, root):
    assert name in ['lfw', 'cfp_fp', 'agedb', 'calfw', 'cplfw']
    imgs, labels = eval_base.load_bin(f'{root}/{name}.bin')
    acc, std, _, bad_case = eval_base.perform_eval(model, imgs, labels)
    logging.info('%s: %1.5f+-%1.5f' % (name, acc, std))
    np.save(f'{LOG_ROOT}/{name}_bad_case', bad_case)
    return acc

def evaluate_lfw(model, cfg):
    LOG_ROOT = f"{cfg['LOG_ROOT']}/base"
    os.makedirs(LOG_ROOT, exist_ok=True)
    acc = evaluate_base(model, 'lfw', LOG_ROOT, cfg['BASE_ROOT'])
    if acc < 1:
        acc *= 100
    return acc

def evaluate_cfp_fp(model, cfg):
    LOG_ROOT = f"{cfg['LOG_ROOT']}/base"
    os.makedirs(LOG_ROOT, exist_ok=True)
    acc = evaluate_base(model, 'cfp_fp', LOG_ROOT, cfg['BASE_ROOT'])
    if acc < 1:
        acc *= 100
    return acc

def evaluate_agedb(model, cfg):
    LOG_ROOT = f"{cfg['LOG_ROOT']}/base"
    os.makedirs(LOG_ROOT, exist_ok=True)
    acc = evaluate_base(model, 'agedb', LOG_ROOT, cfg['BASE_ROOT'])
    if acc < 1:
        acc *= 100
    return acc

def evaluate_calfw(model, cfg):
    LOG_ROOT = f"{cfg['LOG_ROOT']}/base"
    os.makedirs(LOG_ROOT, exist_ok=True)
    acc = evaluate_base(model, 'calfw', LOG_ROOT, cfg['BASE_ROOT'])
    if acc < 1:
        acc *= 100
    return acc

def evaluate_cplfw(model, cfg):
    LOG_ROOT = f"{cfg['LOG_ROOT']}/base"
    os.makedirs(LOG_ROOT, exist_ok=True)
    acc = evaluate_base(model, 'cplfw', LOG_ROOT, cfg['BASE_ROOT'])
    if acc < 1:
        acc *= 100
    return acc

def evaluate_ijbb(model, cfg):
    LOG_ROOT = f"{cfg['LOG_ROOT']}/ijbb"
    os.makedirs(LOG_ROOT, exist_ok=True)
    table, tpr = eval_ijb.perform_eval(model, 'IJBB', LOG_ROOT, cfg['IJB_ROOT'])
    logging.info(table)
    if tpr < 1:
        tpr *= 100
    return tpr

def evaluate_ijbc(model, cfg):
    LOG_ROOT = f"{cfg['LOG_ROOT']}/ijbc"
    os.makedirs(LOG_ROOT, exist_ok=True)
    table, tpr = eval_ijb.perform_eval(model, 'IJBC', LOG_ROOT, cfg['IJB_ROOT'])
    logging.info(table)
    if tpr < 1:
        tpr *= 100
    return tpr

def evaluate_megaface(model, cfg):
    LOG_ROOT = f"{cfg['LOG_ROOT']}/megaface"
    os.makedirs(LOG_ROOT, exist_ok=True)
    acc, tprs, table = eval_megaface.preform_eval(model, LOG_ROOT, cfg['MEGA_ROOT'])
    logging.info(table)
    logging.info(f'rank 1: {acc}')
    logging.info(f'tar@fpr=1e-6: {tprs[5]}')
    if tprs[5] < 1:
        tprs[5] *= 100
    return tprs[5]

def evaluate_all(cfg):
    root = cfg['LOG_ROOT']
    os.makedirs(root, exist_ok=True)
    init_logging(root)
    save_config(root, cfg)

    model = eval(cfg['MODEL_TYPE'])([112, 112])
    model.load_state_dict(torch.load(cfg['MODEL_PATH']))
    model.eval().cuda(cfg['GPU'])

    eval_list = cfg['EVAL_LIST']
    eval_list = [e.lower().replace('-', '_') for e in eval_list]
    results = {}
    for e in eval_list:
        logging.info(f'evaluating {e}...')
        result = eval(f'evaluate_{e}')(model, cfg)
        results[e] = result

    with open(f'{root}/resutlts.txt', 'w') as f:
        for name, result in results.items():
            f.write(f'{name}: {result:.3f}\n')

    return results

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='FR-Evaluation')
    args.add_argument('--opt', default='eval', type=str)
    cfg = configurations[args.parse_args().opt]
    evaluate_all(cfg)