export CUDA_VISIBLE_DEVICES=5
#!/bin/bash
# 定义 ema_decay 值
#ema_decays=(0.991 0.992 0.993 0.994 0.995 0.996 0.997 0.998 0.999 0.9995 0.9999)
ema_decays=(0.9908)
#ema_decays=(0.9906 0.9908)
# 循环遍历每个 ema_decay 值并运行训练脚本
for ema_decay in "${ema_decays[@]}"; do
    # 动态生成 output_dir
    output_dir="work_dirs/real_knowledge_kd_stu_flickr30k_tea_gref_umd_r101_emadecay${ema_decay//./}"

    python -m torch.distributed.launch --nproc_per_node=1 --master_port=29506 --use_env train.py \
        --config configs/VLTVG_R101_flickr.py \
        --dataset mvtec_anomaly_detection \
        --checkpoint_best \
        --checkpoint_step 5 \
        --lr_drop 60 \
        --epochs 100 \
        --batch_size 4 \
        --ema_decay $ema_decay \
        --defined_split deblurred \
        --resume /home/cike/workspace/data/model_zoo/VLTVG_R101_flickr30k.pth \
        --resume_teacher /home/cike/workspace/data/model_zoo/VLTVG_R101_gref_umd.pth \
        --output_dir $output_dir \
        --ema \
        --encoder_distillation \
        --update_ema_teacher_model \
        --detection_head_distillation 
done