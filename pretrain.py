import os
import argparse

import torch
from torch import nn
import torch.optim as optim
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.nn.functional as F

from utils.utils_callback import *
from utils.utils_logging import save_config, init_logging
from utils.utils_seed import set_seed
from utils.utils_backbone import separate_irse_bn_paras, adjust_learning_rate

from dataset.datasets import *
from head.logits import *
from backbone.model_irse import *
from config import configurations


def main(opt):
    cfg = configurations[opt]
    root = f'runs/{cfg["EXP_NAME"]}'

    # create model root & save config
    os.makedirs(root, exist_ok=True)
    save_config(root, cfg)

    # start ddp training
    n_gpus = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, cfg, root))

def main_worker(gpu, n_gpus, cfg, root):
    set_seed(cfg['SEED'])
    init_logging(root)
    if gpu == 0:
        logging.info(f'results will be saved in {os.path.abspath(root)}')

    world_size = torch.cuda.device_count()
    dist.init_process_group(backend='nccl', 
        init_method='tcp://localhost:8800', world_size=world_size, rank=gpu)

    # prepare training data & testing data
    total_batch_size = cfg['BATCH_SIZE']
    batch_size = total_batch_size // n_gpus
    n_workers = (cfg['NUM_WORKERS'] + n_gpus - 1) // n_gpus
    train_transform = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(), # 1. normalize to [0.0, 1.0]; 2. [H, W, C] or PIL to [C, H, W]
       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
       ])

    dataset_train = FaceDataset(cfg["TRAIN_DATA_ROOT"], train_transform)
    dataset_train.scores = np.random.rand(len(dataset_train.samples))
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, 
        shuffle=False, num_workers=n_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    n = len(dataset_train)
    nc = len(dataset_train.classes)
    if gpu == 0:
        logging.info(f"number of training images: {n}")
        logging.info(f"number of training classes: {nc}")
    
    # init backbone etc.
    torch.cuda.set_device(gpu)
    backbone = eval(cfg['BACKBONE_NAME'])(cfg['INPUT_SIZE'])
    head = eval(cfg['HEAD_NAME'])(cfg['EMBEDDING_SIZE'], nc)

    paras_bn, paras_wo_bn = separate_irse_bn_paras(backbone)
    optimizer = optim.SGD([{'params': paras_wo_bn + list(head.parameters()), 'weight_decay': cfg['WEIGHT_DECAY']}, 
        {'params': paras_bn}], lr=cfg['LR'], momentum=cfg['MOMENTUM'])
    
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    backbone.cuda(gpu)
    head.cuda(gpu)

    backbone = nn.parallel.DistributedDataParallel(backbone, device_ids=[gpu])
    head = nn.parallel.DistributedDataParallel(head, device_ids=[gpu])

    # prepare callback
    start_epoch = 0
    global_step = 0
    total_step = len(train_loader) * (cfg['END_EPOCH'] - start_epoch + 1)
    callback_logging = CallBackLogging(100, gpu, total_step, total_batch_size, global_step)
    crucial_epochs = [s - 1 for s in cfg['STAGES']] + [cfg['END_EPOCH']]
    callback_verification = CallBackVerification(10000, crucial_epochs, len(train_loader), gpu)
    callback_checkpoint = CallBackModelCheckpoint(-1, crucial_epochs, len(train_loader), gpu, output=root)
    callback_verification.init_dataset(data_dir=cfg['EVAL_DATA_ROOT'], val_targets=cfg['EVAL_TARGETS'])

    # start training
    ce_loss_meter = AverageMeter()
    cl_loss_meter = AverageMeter()
    backbone.train()
    head.train()
    for epoch in range(start_epoch, cfg['END_EPOCH'] + 1):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, cfg['LR'], cfg['STAGES'])

        for imgs, labels, indexes in train_loader:
            imgs = imgs.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)

            embeddings = backbone(imgs)
            logits, scores, dict_indexes = head(embeddings, labels, nc=1)
            ce_loss = criterion(logits, labels)
            optimizer.zero_grad()
            ce_loss.backward()

            dataset_train.update_scores(scores.cpu().numpy(), indexes)
            scores_node = torch.tensor(dataset_train.scores, dtype=torch.float32).cuda(gpu) / world_size
            dist.all_reduce(scores_node, op=dist.ReduceOp.SUM)
            dataset_train.scores = scores_node.cpu().numpy()

            optimizer.step()

            # callback
            ce_loss_meter.update(ce_loss.detach().cpu().item(), 1)
            callback_logging(epoch, global_step, ce_loss_meter, cl_loss_meter, optimizer.param_groups[0]['lr'])
            callback_verification(backbone, epoch, global_step)
            callback_checkpoint(epoch, global_step, backbone, head, dataset_train.scores)

            global_step += 1
        

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='FaceLab')
    args.add_argument('--opt', default='default', type=str)
    main(args.parse_args().opt)