import os
import time
import torch
import logging
import numpy as np

from eval import eval_base
from eval import eval_ijb
from utils.utils_logging import AverageMeter


class CallBack():
    def __init__(self, freq_step, freq_epoch, steps_per_epoch, rank, time_from_zero=False):
        self.freq_step = freq_step
        self.freq_epoch = freq_epoch
        self.steps_per_epoch = steps_per_epoch
        self.rank = rank
        self.time_from_zero = time_from_zero

    def time_to_call(self, global_step):
        if type(self.freq_step) == list:
            step_flag = global_step in self.freq_step
        else:
            step_flag = self.freq_step > 0 and \
                        global_step + int(self.time_from_zero) > 0 and \
                        global_step % self.freq_step == 0

        if type(self.freq_epoch) == list:
            epoch_flag = (global_step + 1) % self.steps_per_epoch == 0 and \
                         (global_step // self.steps_per_epoch) in self.freq_epoch
        else:
            epoch_flag = self.freq_epoch > 0 and \
                         global_step % (self.steps_per_epoch * self.freq_epoch) == 0
        
        return self.rank == 0 and (step_flag or epoch_flag)


class CallBackVerification(CallBack):
    def __init__(self, freq_step, freq_epoch, steps_per_epoch, rank, image_size=(112, 112)):
        super(CallBackVerification, self).__init__(freq_step, freq_epoch, steps_per_epoch, rank)
        self.highest_acc_list = []
        self.ver_list = []
        self.ver_name_list = []
        self.image_size = image_size

    def ver_test(self, backbone, epoch, global_step):
        for i in range(len(self.ver_list)):
            acc, std, xnorm, _ = eval_base.perform_eval(backbone, self.ver_list[i][0], self.ver_list[i][1])
            logging.info('[%s][%d][%dk]XNorm: %f' % (self.ver_name_list[i], epoch, global_step//1000, xnorm))
            logging.info('[%s][%d][%dk]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], epoch, global_step//1000, acc, std))
            if acc > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc
            logging.info(
                '[%s][%d][%dk]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], epoch, global_step//1000, self.highest_acc_list[i]))

    def init_dataset(self, data_dir, val_targets):
        if self.rank == 0:
            for name in val_targets:
                self.ver_name_list.append(name)
                self.highest_acc_list.append(0.0)
                path = os.path.join(data_dir, name + ".bin")
                if os.path.exists(path):
                    data_set = eval_base.load_bin(path, self.image_size)
                    self.ver_list.append(data_set)

    def __call__(self, backbone, epoch, global_step):
        if self.time_to_call(global_step):
            backbone.eval()
            self.ver_test(backbone, epoch, global_step)
            backbone.train()


class CallBackVerificationIJB(CallBack):
    def __init__(self, freq_step, freq_epoch, steps_per_epoch, rank, data_root, output, image_size=(112, 112)):
        super(CallBackVerificationIJB, self).__init__(freq_step, freq_epoch, steps_per_epoch, rank)
        self.data_root = data_root
        self.output = output

    def __call__(self, backbone, epoch, global_step):
        if self.time_to_call(global_step):
            backbone.eval()
            name = f'{epoch}_{global_step//1000}k'
            os.makedirs(f'{self.output}/{name}/ijbb', exist_ok=True)
            eval_ijb.perform_eval(backbone, 'IJBB', f'{self.output}/{name}/ijbb', 
                self.data_root)
            os.makedirs(f'{self.output}/{name}/ijbc', exist_ok=True)
            eval_ijb.perform_eval(backbone, 'IJBC', f'{self.output}/{name}/ijbc', 
                self.data_root)
            backbone.train()


class CallBackLogging(CallBack):
    def __init__(self, frequent, rank, total_step, total_batch_size, start_step=0):
        super(CallBackLogging, self).__init__(frequent, -1, -1, rank, True)
        self.time_start = time.time()
        self.frequent = frequent
        self.total_step = total_step
        self.start_step = start_step
        self.total_batch_size = total_batch_size
        self.init = False
        self.tic = 0

    def __call__(self, epoch, global_step, ce_loss_meter, cl_loss_meter, learning_rate):
        if self.time_to_call(global_step):
            if self.init:
                speed = self.frequent * self.total_batch_size / (time.time() - self.tic)

                time_now = (time.time() - self.time_start) / 3600
                time_total = time_now / (global_step - self.start_step + 1) * self.total_step
                time_for_end = time_total - time_now

                msg = "Speed %.2f samples/sec   CE_Loss %.4f   CL_Loss: %.4f   LR %.4f   Epoch: %d   Global Step: %d   " \
                      "Required: %.1f hours" % (speed, ce_loss_meter.avg, cl_loss_meter.avg, learning_rate, epoch, global_step, time_for_end)

                logging.info(msg)
                ce_loss_meter.reset()
                cl_loss_meter.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()


class CallBackModelCheckpoint(CallBack):
    def __init__(self, freq_step, freq_epoch, steps_per_epoch, rank, output="./"):
        super(CallBackModelCheckpoint, self).__init__(freq_step, freq_epoch, steps_per_epoch, rank)
        self.output = output

    def __call__(self, epoch, global_step, backbone, head=None, scores=None):
        if self.time_to_call(global_step):
            suffix = f'{epoch}_{global_step//1000}k'
            path_backbone = os.path.join(self.output, f"backbone_{suffix}.pt")
            path_head = os.path.join(self.output, f"head_{suffix}.pt")
            path_scores = os.path.join(self.output, f'scores_{suffix}')

            torch.save(backbone.module.state_dict(), path_backbone)
            logging.info("model saved in '{}'".format(path_backbone))

            if head is not None:
                torch.save({'EPOCH': epoch, "GLOBAL_STEP": global_step, 
                    'HEAD':head.module.state_dict()}, path_head)

            if scores is not None:
                np.save(path_scores, scores)


    