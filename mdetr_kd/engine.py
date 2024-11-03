# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Dict, Iterable, Optional

import torch
import torch.nn
import torch.optim
import torch.nn.functional as F

import util.dist as dist
from datasets.clevrref import ClevrRefEvaluator
from datasets.coco_eval import CocoEvaluator
from datasets.flickr_eval import FlickrEvaluator
from datasets.phrasecut_eval import PhrasecutEvaluator
from datasets.refexp import RefExpEvaluator
from util.metrics import MetricLogger, SmoothedValue
from util.misc import targets_to
from util.optim import adjust_learning_rate, update_ema


def train_one_epoch(
    model: torch.nn.Module,
    teacher_model: Optional[torch.nn.Module],
    criterion: Optional[torch.nn.Module],
    contrastive_criterion: Optional[torch.nn.Module],
    qa_criterion: Optional[torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    max_norm: float = 0,
    model_ema: Optional[torch.nn.Module] = None,
    teacher_model_ema: Optional[torch.nn.Module] = None,
    pretrained_dict: Optional[Dict] = None,
):
    model.train()
    if criterion is not None:
        criterion.train()
    if contrastive_criterion is not None:
        contrastive_criterion.train()
    if qa_criterion is not None:
        qa_criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_text_encoder", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    num_training_steps = int(len(data_loader) * args.epochs)
    for i, batch_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        curr_step = epoch * len(data_loader) + i
        samples = batch_dict["samples"].to(device)
        positive_map = batch_dict["positive_map"].to(device) if "positive_map" in batch_dict else None
        targets = batch_dict["targets"]
        answers = {k: v.to(device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else None
        captions = [t["caption"] for t in targets]

        targets = targets_to(targets, device)
        loss_dict = {}
        memory_cache = None
        if args.masks:
            outputs = model(samples, captions)
        else:
            memory_cache = model(samples, captions, encode_and_save=True)
            outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)
            if teacher_model is not None:
                with torch.no_grad():
                    if args.encoder_distillation or args.detection_head_distillation or args.text_distillation:
                        teacher_memory_cache = teacher_model(samples, captions, encode_and_save=True)
                    
                    if args.encoder_distillation:
                        distillation_loss_dict = distillation_loss(teacher_memory_cache["img_memory"], memory_cache["img_memory"], temperature=args.distillation_temperature, \
                            alpha=args.distillation_alpha, scaled_weight=args.distillation_scaled_weight)
                        loss_dict.update(distillation_loss_dict)
                    #text_memory_loss = transformer_distillation_loss(memory_cache["text_attn"], teacher_memory_cache["text_attn"])
                    #loss_dict["text_memory_loss"] = text_memory_loss
                    if args.text_distillation:
                        text_distillation_loss = visual_feature_distillation_loss(memory_cache["text_memory"], teacher_memory_cache["text_memory"],temperature=10.0)
                        loss_dict["text_distillation_loss"] = text_distillation_loss
                    if args.detection_head_distillation:
                        teacher_outputs = teacher_model(samples, captions, encode_and_save=False, memory_cache=teacher_memory_cache)
                        detection_head_loss = detection_head_distillation_loss(outputs["pred_logits"], teacher_outputs["pred_logits"], outputs["pred_boxes"], teacher_outputs["pred_boxes"],temperature=10.0)
                        loss_dict["detection_head_loss"] = detection_head_loss
        
        if criterion is not None:
            
            loss_dict.update(criterion(outputs, targets, positive_map))

        if contrastive_criterion is not None:
            assert memory_cache is not None
            contrastive_loss = contrastive_criterion(memory_cache["text_pooled_op"], memory_cache["img_pooled_op"])
            loss_dict["contrastive_loss"] = contrastive_loss

        if qa_criterion is not None:
            answer_losses = qa_criterion(outputs, answers)
            loss_dict.update(answer_losses)

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
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

        adjust_learning_rate(
            optimizer,
            epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )
        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)
        if teacher_model_ema is not None and args.update_ema_infer_teacher_model:
            update_ema(model, teacher_model_ema.module, args.ema_decay)
        if args.update_teacher_ema_infer_model_from_model_ema:
            update_ema(model_ema, teacher_model_ema.module, args.ema_decay)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
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

def detection_head_distillation_loss(student_logits, teacher_logits, student_boxes, teacher_boxes, temperature=2.0, alpha=2.0):
    cls_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction='batchmean')
    box_loss = F.mse_loss(student_boxes, teacher_boxes)
    return cls_loss + box_loss*temperature

def visual_feature_distillation_loss(student_features, teacher_features, temperature=2.0, alpha=2.0):
    return F.mse_loss(student_features, teacher_features)*temperature

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    contrastive_criterion: Optional[torch.nn.Module],
    qa_criterion: Optional[torch.nn.Module],
    postprocessors: Dict[str, torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader,
    evaluator_list,
    device: torch.device,
    args,
):
    model.eval()
    if criterion is not None:
        criterion.eval()
    if contrastive_criterion is not None:
        contrastive_criterion.eval()
    if qa_criterion is not None:
        qa_criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for batch_dict in metric_logger.log_every(data_loader, 10, header):
        samples = batch_dict["samples"].to(device)
        positive_map = batch_dict["positive_map"].to(device) if "positive_map" in batch_dict else None
        targets = batch_dict["targets"]
        answers = {k: v.to(device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else None
        captions = [t["caption"] for t in targets]

        targets = targets_to(targets, device)

        memory_cache = None
        if args.masks:
            outputs = model(samples, captions)
        else:
            memory_cache = model(samples, captions, encode_and_save=True)
            outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)

        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, positive_map))

        if contrastive_criterion is not None:
            assert memory_cache is not None
            contrastive_loss = contrastive_criterion(memory_cache["text_pooled_op"], memory_cache["img_pooled_op"])
            loss_dict["contrastive_loss"] = contrastive_loss

        if qa_criterion is not None:
            answer_losses = qa_criterion(outputs, answers)
            loss_dict.update(answer_losses)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )

        if not args.no_detection:
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors["bbox"](outputs, orig_target_sizes)
            if "segm" in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)

            flickr_res = [] if "flickr_bbox" in postprocessors.keys() else None
            if "flickr_bbox" in postprocessors.keys():
                image_ids = [t["original_img_id"] for t in targets]
                sentence_ids = [t["sentence_id"] for t in targets]
                items_per_batch_element = [t["nb_eval"] for t in targets]
                positive_map_eval = batch_dict["positive_map_eval"].to(device)
                flickr_results = postprocessors["flickr_bbox"](
                    outputs, orig_target_sizes, positive_map_eval, items_per_batch_element
                )
                assert len(flickr_results) == len(image_ids) == len(sentence_ids)
                for im_id, sent_id, output in zip(image_ids, sentence_ids, flickr_results):
                    flickr_res.append({"image_id": im_id, "sentence_id": sent_id, "boxes": output})

            phrasecut_res = None
            if "phrasecut" in postprocessors.keys():
                phrasecut_res = postprocessors["phrasecut"](results)
                assert len(targets) == len(phrasecut_res)
                for i in range(len(targets)):
                    phrasecut_res[i]["original_id"] = targets[i]["original_id"]
                    phrasecut_res[i]["task_id"] = targets[i]["task_id"]

            res = {target["image_id"].item(): output for target, output in zip(targets, results)}

            for evaluator in evaluator_list:
                if isinstance(evaluator, FlickrEvaluator):
                    evaluator.update(flickr_res)
                elif isinstance(evaluator, PhrasecutEvaluator):
                    evaluator.update(phrasecut_res)
                else:
                    evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()

    refexp_res = None
    flickr_res = None
    phrasecut_res = None
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            evaluator.accumulate()
            evaluator.summarize()

        elif isinstance(evaluator, (RefExpEvaluator, ClevrRefEvaluator)):
            refexp_res = evaluator.summarize()
        elif isinstance(evaluator, FlickrEvaluator):
            flickr_res = evaluator.summarize()
        elif isinstance(evaluator, PhrasecutEvaluator):
            phrasecut_res = evaluator.summarize()

    # accumulate predictions from all images

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            if "bbox" in postprocessors.keys():
                stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()
            if "segm" in postprocessors.keys():
                stats["coco_eval_masks"] = evaluator.coco_eval["segm"].stats.tolist()

    if refexp_res is not None:
        stats.update(refexp_res)

    if flickr_res is not None:
        stats["flickr"] = flickr_res

    if phrasecut_res is not None:
        stats["phrasecut"] = phrasecut_res

    return stats
