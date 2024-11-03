# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist
from typing import Iterable, Tuple, Dict
from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
import torch.nn.functional as F

def train_one_epoch(args, model: torch.nn.Module, model_ema: torch.nn.Module, teacher_model: torch.nn.Module, teacher_model_ema: torch.nn.Module, 
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, device: torch.device, 
                    epoch: int, max_norm: float = 0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        img_data, text_data, target = batch

        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        # model forward
        outputs = model(img_data, text_data)

        loss_dict = loss_utils.trans_vg_loss(outputs['pred_boxes'], target)

        if teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(img_data, text_data)
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

        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v
                                      for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

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
        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

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


@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()
    iou_thrs = torch.as_tensor([0.5 + 0.05 * i for i in range(0,9)], device=device)
    accum_acc = 0
    accum_iou = 0
    accum_sample = 0
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        img_data, text_data, target = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        
        outputs = model(img_data, text_data)
        pred_boxes = outputs['pred_boxes']
        miou, accu = eval_utils.trans_vg_eval_val(pred_boxes, target)
        sum_iou = miou.sum()
        num_acc = (miou[:, None] > iou_thrs[None]).sum(dim=0)
        num_sample = torch.as_tensor(batch_size, device=img_data.tensors.device)
        accum_acc += num_acc
        accum_iou += sum_iou
        accum_sample += num_sample
        
        metric_logger.update_v2('miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('accu', accu, batch_size)
    if utils.get_world_size() > 1:
        dist.all_reduce(accum_acc)
        dist.all_reduce(accum_iou)
        dist.all_reduce(accum_sample)
    acc = accum_acc / accum_sample
    miou = accum_iou.item() / accum_sample
    val_acc = {f'Acc@{t:.2f}': a.item() for t, a in zip(iou_thrs, acc)}
    val_acc['mIoU'] = miou
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats, val_acc


@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        outputs = model(img_data, text_data)
        output = outputs['pred_boxes']
        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)
    
    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    
    return accuracy
        