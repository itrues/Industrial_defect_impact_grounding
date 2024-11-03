
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env  --master_port=29504 main_test.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 1  \
    --ema \
    --text_encoder_lr 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 10 \
    --eval \
    --schedule multistep \
    --backbone "timm_tf_efficientnet_b5_ns" \
    --output_dir ./results/logs_pretrained_EB5_multistep_masked_mvtec_anomaly_detection_ratio_8_seed_65_mixed_deblurred \
    --resume '/home/cike/workspace/github/fromgithub/mdetr/results/logs_pretrained_EB5_multistep_masked_mvtec_anomaly_detection_ratio_8_seed_65_mixed_blurred/checkpoint.pth'

