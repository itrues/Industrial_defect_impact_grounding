export CUDA_VISIBLE_DEVICES=6

# ReferItGame
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29506 --use_env train.py \
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
    --ema_decay 0.999 \
    --epochs 120 \
    --output_dir outputs/real_knowledge_kd_gref_umd_r101_emadecay0999 \
    --ema \
    --encoder_distillation \
    --update_ema_teacher_model \
    --detection_head_distillation \


# # RefCOCO
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --output_dir outputs/refcoco_r50 


# # RefCOCO+
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --output_dir outputs/refcoco_plus_r50 --epochs 180 --lr_drop 120


# # RefCOCOg g-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone resnet50 --detr_model ./checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50


# # RefCOCOg umd-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone resnet50 --detr_model ./checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_usplit_r50
