export CUDA_VISIBLE_DEVICES=7

# ReferItGame
#!/bin/bash

# 定义 ema_decay 值
ema_decays=(0.991 0.992 0.993 0.994 0.995 0.996 0.997 0.998 0.9995 0.9999)

# 循环遍历每个 ema_decay 值并运行训练脚本
for ema_decay in "${ema_decays[@]}"; do
    # 动态生成 output_dir
    output_dir="outputs/real_knowledge_kd_gref_umd_r101_emadecay${ema_decay//./}"

    python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 --use_env train.py \
        --batch_size 4 \
        --lr_bert 0.00001 \
        --aug_crop \
        --aug_scale \
        --aug_translate \
        --backbone resnet101 \
        --resume /home/cike/workspace/data/model_zoo/transvg/R-101/TransVG_R101_gref_umd.pth \
        --resume_teacher /home/cike/workspace/data/model_zoo/transvg/R-101/TransVG_R101_gref.pth \
        --bert_enc_num 12 \
        --detr_enc_num 6 \
        --dataset mvtec_anomaly_detection \
        --max_query_len 40 \
        --ema_decay $ema_decay \
        --epochs 120 \
        --output_dir $output_dir \
        --ema \
        --encoder_distillation \
        --update_ema_teacher_model \
        --detection_head_distillation
done


# # RefCOCO
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --output_dir outputs/refcoco_r50 


# # RefCOCO+
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --output_dir outputs/refcoco_plus_r50 --epochs 180 --lr_drop 120


# # RefCOCOg g-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone resnet50 --detr_model ./checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50


# # RefCOCOg umd-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone resnet50 --detr_model ./checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_usplit_r50
