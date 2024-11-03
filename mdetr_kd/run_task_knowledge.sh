
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --use_env  --master_port=29506 main.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 1  \
    --load /home/cike/workspace/data/refer/mdetr/mdetr/checkpoint/pretrained_resnet101_checkpoint.pth \
    --teacher_model_load /home/cike/workspace/data/refer/mdetr/mdetr/checkpoint/refcocog_resnet101_checkpoint.pth \
    --text_encoder_lr 1e-5 \
    --lr_backbone 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 40 \
    --epochs 100 \
    --ema \
    --accumulate_update_weights \
    --encoder_distillation \
    --update_teacher_ema_infer_model_from_model_ema \
    --schedule multistep \
    --backbone "resnet101" \
    --output_dir ./logs/logs_refcocog_resnet101_knowledge_mvtec_anomaly_detection_ratio_8_seed_65_epoch_100-v6
    #--resume '/home/cike/workspace/github/fromgithub/mdetr/results/logs_pretrained_EB5_multistep_masked_ratio_8_seed_65_epoch_60_blured/BEST_checkpoint.pth'



CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --use_env  --master_port=29506 main.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 1  \
    --load /home/cike/workspace/data/refer/mdetr/mdetr/checkpoint/pretrainedrefcocog_EB3_checkpoint.pth \
    --teacher_model_load /home/cike/workspace/data/refer/mdetr/mdetr/checkpoint/refcocog_EB3_checkpoint.pth \
    --text_encoder_lr 1e-5 \
    --lr_backbone 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 40 \
    --epochs 100 \
    --ema \
    --accumulate_update_weights \
    --encoder_distillation \
    --update_teacher_ema_infer_model_from_model_ema \
    --schedule multistep \
    --backbone "timm_tf_efficientnet_b3_ns" \
    --output_dir ./logs/logs_refcocog_EB3_knowledge_mvtec_anomaly_detection_ratio_8_seed_65_65_epoch_100-v6
    #--resume '/home/cike/workspace/github/fromgithub/mdetr/results/logs_pretrained_EB5_multistep_masked_ratio_8_seed_65_epoch_60_blured/BEST_checkpoint.pth'

