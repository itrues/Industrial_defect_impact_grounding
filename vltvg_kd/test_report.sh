export CUDA_VISIBLE_DEVICES=5

python -m torch.distributed.launch --nproc_per_node=1 --master_port=29509 --use_env train.py \
    --config configs/VLTVG_R101_flickr.py \
    --dataset mvtec_anomaly_detection \
    --checkpoint_best \
    --checkpoint_step 5 \
    --lr_drop 60 \
    --epochs 120 \
    --batch_size_val 1 \
    --eval \
    --ema_decay 0.9995 \
    --defined_split deblurred \
    --resume work_dirs/VLTVG_R101_gref_umd_knowledge_distillation_batch_4_epoch_120_emadecay_0.995/checkpoint_best_acc.pth \
    --output_dir results/ \
    --encoder_distillation \
    --update_ema_teacher_model \
    --detection_head_distillation 
