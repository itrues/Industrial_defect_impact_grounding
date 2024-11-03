import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from util import box_ops

import logging
import torch.distributed as dist
import time
import datetime
from tqdm import tqdm
from typing import Dict, Iterable, Optional
import torch.nn.functional as F

class data_prefetcher():
    def __init__(self, loader, device):
        self.length = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.preload()

    def preload(self):
        try:
            samples, targets, image_path = next(self.loader)
            self.next_img, self.next_mask = samples.decompose()
            self.image_path = image_path
            self.next_target = targets
        except StopIteration:
            self.next_img = self.next_mask = self.next_target = self.image_path = None
            return
        with torch.cuda.stream(self.stream):
            self.next_img = self.next_img.to(self.device, non_blocking=True)
            self.next_mask = self.next_mask.to(self.device, non_blocking=True)
            tensor_dict = self.next_target.tensor_dict
            self.next_target.tensor_dict = {k: tensor_dict[k].to(self.device, non_blocking=True) for k in tensor_dict}

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img, mask, target, image_path = self.next_img, self.next_mask, self.next_target, self.image_path
        self.preload()
        return img, mask, target, image_path

    def __next__(self):
        img, mask, target, image_path = self.next()
        if img == None:
            raise StopIteration
        return img, mask, target, image_path

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

def update_ema(model, model_ema, decay):
    """Apply exponential moving average update.

    The  weights are updated in-place as follow:
    w_ema = w_ema * decay + (1 - decay) * w
    Args:
        model: active model that is being optimized
        model_ema: running average model
        decay: exponential decay parameter
    """
    with torch.no_grad():
        if hasattr(model, "module"):
            # unwrapping DDP
            model = model.module
        msd = model.state_dict()
        for k, ema_v in model_ema.state_dict().items():
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)



def train_one_epoch(model: torch.nn.Module, model_ema: torch.nn.Module, teacher_model: torch.nn.Module, teacher_model_ema: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, epochs: int, max_norm: float = 0, args=None):
    model.train()
    criterion.train()
    logger = logging.getLogger("train")
    metric_logger = utils.MetricLogger(delimiter="  ")

    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')
    header = 'Epoch [{epoch}][{iter}/{max_iter}]'

    max_iter = len(data_loader)
    end = time.time()

    prefetcher = data_prefetcher(data_loader, device)
    img, mask, target, _ = prefetcher.next()
    iteration = 0
    while img is not None:
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        #print(target_dict['image_path'])
        iteration = iteration + 1
        data_time.update(time.time() - end)

        outputs = model(img, mask, word_id, word_mask)
        loss_dict = criterion(outputs, target_dict)
        if teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(img, mask, word_id, word_mask)
                if args.encoder_distillation:
                    distillation_loss_dict = distillation_loss(outputs["vl_feat"], teacher_outputs["vl_feat"], temperature=args.distillation_temperature, \
                            alpha=args.distillation_alpha, scaled_weight=args.distillation_scaled_weight)
                    loss_dict.update(distillation_loss_dict)
                #text_memory_loss = transformer_distillation_loss(memory_cache["text_attn"], teacher_memory_cache["text_attn"])
                #loss_dict["text_memory_loss"] = text_memory_loss
                if args.text_distillation:
                    text_distillation_loss = visual_feature_distillation_loss(outputs["text_feat"], teacher_outputs["text_feat"],temperature=10.0)
                    loss_dict["text_distillation_loss"] = text_distillation_loss
                if args.detection_head_distillation:
                    detection_head_loss = detection_head_distillation_loss( outputs["pred_boxes"], teacher_outputs["pred_boxes"],temperature=10.0)
                    loss_dict["detection_head_loss"] = detection_head_loss
        
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)
        if teacher_model_ema is not None:
            update_ema(model, teacher_model_ema.module, args.ema_decay)
        iter_time.update(time.time() - end)
        end = time.time()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

        if iteration % 50 == 0 or iteration == max_iter:
            eta_seconds = iter_time.global_avg * (max_iter - iteration + max_iter * (epochs-epoch-1))
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                metric_logger.delimiter.join(
                    [header,
                     "lr: {lr}",
                     "eta: {eta}",
                     "time: {time}",
                     "data: {data}",
                     "memory: {memory:.0f}",
                     "{meters}"
                     ]
                ).format(
                    epoch=epoch+1, iter=iteration, max_iter=max_iter,
                    lr=optimizer.param_groups[0]["lr"],
                    eta=eta_string,
                    time=str(iter_time),
                    data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                    meters=str(metric_logger)
                ))

        img, mask, target, _ = prefetcher.next()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def distillation_loss(
    teacher_outputs: Dict[str, torch.Tensor],
    student_outputs: Dict[str, torch.Tensor],
    temperature: float=2.0,
    alpha: float=2.0,
    scaled_weight: float=5.0,
):
    loss_dict = {}
    """kl_loss = F.kl_div(
        F.log_softmax(student_outputs / temperature, dim=1),
        F.softmax(teacher_outputs / temperature, dim=1),
        reduction="sum",
    )"""
    mse_loss = F.mse_loss(student_outputs, teacher_outputs)
    loss_dict["mse_loss"] = mse_loss*scaled_weight

    return loss_dict

def transformer_distillation_loss(student_attn, teacher_attn,temperature=2.0, alpha=2.0):
    return F.mse_loss(student_attn, teacher_attn)

def detection_head_distillation_loss( student_boxes, teacher_boxes, temperature=2.0, alpha=2.0):
    box_loss = F.mse_loss(student_boxes, teacher_boxes)
    return box_loss*temperature

def visual_feature_distillation_loss(student_features, teacher_features, temperature=2.0, alpha=2.0):
    return F.mse_loss(student_features, teacher_features)*temperature

def train_one_epoch_w_accum(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, epochs: int, max_norm: float = 0):
    model.train()
    criterion.train()
    logger = logging.getLogger("train")
    metric_logger = utils.MetricLogger(delimiter="  ")

    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')
    header = 'Epoch [{epoch}][{iter}/{max_iter}]'

    max_iter = len(data_loader)
    end = time.time()

    prefetcher = data_prefetcher(data_loader, device)
    img, mask, target, _ = prefetcher.next()
    iteration = 0
    while img is not None:
        target_dict = target.tensor_dict
        iteration = iteration + 1
        data_time.update(time.time() - end)

        B = img.shape[0]
        b = B // 2
        loss_dicts = list()
        weight_dict = criterion.weight_dict
        for i in range(2):
            b_img = img[i*b:(i+1)*b]
            b_mask = mask[i*b:(i+1)*b]
            b_target = {k: target_dict[k][i*b:(i+1)*b] for k in target_dict}
            b_word_id, b_word_mask = b_target['word_id'], b_target['word_mask']

            outputs = model(b_img, b_mask, b_word_id, b_word_mask)

            loss_dict = criterion(outputs, b_target)
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) / 2
            losses.backward()
            loss_dicts.append(loss_dict)

        loss_dict_accum_scaled = {k: (loss_dicts[0][k] + loss_dicts[1][k]) * weight_dict[k] / 2
                                    for k in loss_dicts[0].keys() if k in weight_dict}

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced_scaled = utils.reduce_dict(loss_dict_accum_scaled)
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced_scaled)
            sys.exit(1)

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        optimizer.zero_grad()

        iter_time.update(time.time() - end)
        end = time.time()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

        if iteration % 100 == 0 or iteration == max_iter:
            eta_seconds = iter_time.global_avg * (max_iter - iteration + max_iter * (epochs-epoch-1))
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                metric_logger.delimiter.join(
                    [header,
                     "lr: {lr}",
                     "eta: {eta}",
                     "time: {time}",
                     "data: {data}",
                     "memory: {memory:.0f}",
                     "{meters}"
                     ]
                ).format(
                    epoch=epoch+1, iter=iteration, max_iter=max_iter,
                    lr=optimizer.param_groups[0]["lr"],
                    eta=eta_string,
                    time=str(iter_time),
                    data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                    meters=str(metric_logger)
                ))

        img, mask, target = prefetcher.next()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessor, data_loader, device, save_path=''):
    model.eval()
    if criterion:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')

    accum_acc = 0
    accum_iou = 0
    accum_sample = 0
    iou_thrs = torch.as_tensor([0.5 + 0.05 * i for i in range(0,9)], device=device)

    end = time.time()

    all_pred_ious = []
    all_pred_boxes = []
    prefetcher = data_prefetcher(data_loader, device)
    """if os.path.exists('results/error_image_blurred.json'):
        import json
        with open('results/error_image_blurred.json', 'r') as f:
            error_image_list = json.load(f)
    else:
        error_image_list = []"""
    image_path_list = []
    nn = 0
    for iteration, (img, mask, target, image_path) in enumerate(tqdm(prefetcher)):
        
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        gt_bbox = target_dict['orig_bbox']

        data_time.update(time.time() - end)

        outputs = model(img, mask, word_id, word_mask)
        if criterion:
            loss_dict = criterion(outputs, target_dict)
            weight_dict = criterion.weight_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_value = sum(loss_dict_reduced_scaled.values()).item()
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)


        pred_boxes = postprocessor(outputs, target_dict)

        ious = box_ops.box_pair_iou(gt_bbox, pred_boxes)[0]
        sum_iou = ious.sum()
        #print(ious,'oooooooo',nn)
        if ious[0].item() > 0.55:
            nn += 1
            #print(ious,'ooooo',image_path)
            image_path_list.append(image_path[0]) #.replace('blur','deblur')
        num_acc = (ious[:, None] > iou_thrs[None]).sum(dim=0)
        num_sample = torch.as_tensor(img.size(0), device=img.device)
        #if ious[0].item() > 0.8:
            #if image_path in error_image_list:
        #print(ious,gt_bbox, pred_boxes,'sssss',image_path)
        accum_acc += num_acc
        accum_iou += sum_iou
        accum_sample += num_sample

        iter_time.update(time.time() - end)
        end = time.time()

        all_pred_ious.append(ious.view(-1, 1))
        all_pred_boxes.append(pred_boxes)
    with open('results/vltvg_our_image_9.json', 'w') as f:
        import json 
        json.dump(image_path_list, f)
    if save_path:
        torch.save({'pred_boxes': torch.cat(all_pred_boxes, dim=0),
                    'pred_ious': torch.cat(all_pred_ious, dim=0)},
                   save_path + 'pred_boxes')
    # accumulate predictions from all images
    #print(accum_acc, accum_iou, accum_sample,'pssssssssssssspppppppppppp')
    if utils.get_world_size() > 1:
        dist.all_reduce(accum_acc)
        dist.all_reduce(accum_iou)
        dist.all_reduce(accum_sample)

    acc = accum_acc / accum_sample
    miou = accum_iou.item() / accum_sample
    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    val_acc = {f'Acc@{t:.2f}': a.item() for t, a in zip(iou_thrs, acc)}
    val_acc.update({'Mean_iou': miou})
    val_time = {'data_time': data_time.global_avg, 'time': iter_time.global_avg}
    return val_stats, val_acc, val_time