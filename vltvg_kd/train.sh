export CUDA_VISIBLE_DEVICES=3

python -m torch.distributed.launch --nproc_per_node=1 --master_port=29505 --use_env train.py \
    --config configs/VLTVG_R101_flickr.py \
    --dataset mvtec_anomaly_detection \
    --checkpoint_best \
    --checkpoint_step 5 \
    --lr_drop 60 \
    --epochs 100 \
    --batch_size 4 \
    --ema_decay 0.999 \
    --defined_split deblurred \
    --resume /home/cike/workspace/data/model_zoo/VLTVG_R101_flickr30k.pth \
    --resume_teacher /home/cike/workspace/data/model_zoo/VLTVG_R101_gref_umd.pth \
    --output_dir work_dirs/VLTVG_R101_flickr_knowledge_distillation_batch_4_epoch_120_emadecay_0.999 \
    --ema \
    --encoder_distillation \
    --update_ema_teacher_model \
    --detection_head_distillation \
