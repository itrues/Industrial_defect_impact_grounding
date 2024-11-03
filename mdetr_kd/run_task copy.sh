

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --use_env  --master_port=29504 main.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 2  \
    --load /home/cike/workspace/data/refer/mdetr/checkpoint/refcocog_EB3_checkpoint.pth \
    --ema \
    --text_encoder_lr 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 30 \
    --backbone "timm_tf_efficientnet_b3_ns" \
    --schedule linear_with_warmup \
    --output_dir ./results/logs_refcocog_EB3_masked

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --use_env  --master_port=29504 main.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 2  \
    --load /home/cike/workspace/data/refer/mdetr/checkpoint/pretrained_EB3_checkpoint.pth \
    --ema \
    --text_encoder_lr 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 30 \
    --schedule linear_with_warmup \
    --backbone "timm_tf_efficientnet_b3_ns" \
    --output_dir ./results/logs_pretrained_EB3_masked


CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --use_env  --master_port=29504 main.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 2  \
    --load /home/cike/workspace/data/refer/mdetr/checkpoint/pretrained_EB5_checkpoint.pth \
    --ema \
    --text_encoder_lr 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 30 \
    --schedule linear_with_warmup \
    --backbone "timm_tf_efficientnet_b5_ns" \
    --output_dir ./results/logs_pretrained_EB5_masked

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --use_env  --master_port=29504 main.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 3  \
    --load /home/cike/workspace/data/refer/mdetr/checkpoint/pretrained_resnet101_checkpoint.pth \
    --ema \
    --text_encoder_lr 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 30 \
    --schedule linear_with_warmup \
    --backbone "resnet101" \
    --output_dir ./results/logs_pretrained_resnet101_masked

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --use_env  --master_port=29504 main.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 3  \
    --load /home/cike/workspace/data/refer/mdetr/checkpoint/refcocog_resnet101_checkpoint.pth \
    --ema \
    --text_encoder_lr 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 10 \
    --schedule multistep \
    --backbone "resnet101" \
    --output_dir ./results/logs_refcocog_resnet101_multistep_masked

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --use_env  --master_port=29504 main.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 2  \
    --load /home/cike/workspace/data/refer/mdetr/checkpoint/refcocog_EB3_checkpoint.pth \
    --ema \
    --text_encoder_lr 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 10 \
    --schedule multistep \
    --backbone "timm_tf_efficientnet_b3_ns" \
    --output_dir ./results/logs_refcocog_EB3_multistep_masked

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --use_env  --master_port=29504 main.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 2  \
    --load /home/cike/workspace/data/refer/mdetr/checkpoint/pretrained_EB3_checkpoint.pth \
    --ema \
    --text_encoder_lr 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 10 \
    --schedule multistep \
    --backbone "timm_tf_efficientnet_b3_ns" \
    --output_dir ./results/logs_pretrained_EB3_multistep_masked


CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --use_env  --master_port=29504 main.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 2  \
    --load /home/cike/workspace/data/refer/mdetr/checkpoint/pretrained_EB5_checkpoint.pth \
    --ema \
    --text_encoder_lr 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 10 \
    --schedule multistep \
    --backbone "timm_tf_efficientnet_b5_ns" \
    --output_dir ./results/logs_pretrained_EB5_multistep_masked

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --use_env  --master_port=29504 main.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 3  \
    --load /home/cike/workspace/data/refer/mdetr/checkpoint/pretrained_resnet101_checkpoint.pth \
    --ema \
    --text_encoder_lr 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 10 \
    --schedule multistep \
    --backbone "resnet101" \
    --output_dir ./results/logs_pretrained_resnet101_multistep_masked
    


##############################

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env  --master_port=29504 main.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 1  \
    --load /home/cike/workspace/data/refer/mdetr/mdetr/checkpoint/pretrained_EB3_checkpoint.pth \
    --ema \
    --text_encoder_lr 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 10 \
    --schedule multistep \
    --backbone "timm_tf_efficientnet_b3_ns" \
    --output_dir ./results/logs_pretrained_EB3_multistep_knowledge_mvtec_anomaly_detection_ratio_8_seed_65
    #--resume '/home/cike/workspace/github/fromgithub/mdetr/results/logs_pretrained_EB5_multistep_masked_ratio_8_seed_65_epoch_60_blured/BEST_checkpoint.pth'


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env  --master_port=29504 main.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 1  \
    --load /home/cike/workspace/data/refer/mdetr/mdetr/checkpoint/pretrained_EB5_checkpoint.pth \
    --ema \
    --text_encoder_lr 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 10 \
    --schedule multistep \
    --backbone "timm_tf_efficientnet_b5_ns" \
    --output_dir ./results/logs_pretrained_EB5_multistep_knowledge_mvtec_anomaly_detection_ratio_8_seed_65
    #--resume '/home/cike/workspace/github/fromgithub/mdetr/results/logs_pretrained_EB5_multistep_masked_ratio_8_seed_65_epoch_60_blured/BEST_checkpoint.pth'


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env  --master_port=29504 main.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 1  \
    --load /home/cike/workspace/data/refer/mdetr/mdetr/checkpoint/pretrained_resnet101_checkpoint.pth \
    --ema \
    --text_encoder_lr 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 10 \
    --schedule multistep \
    --backbone "resnet101" \
    --output_dir ./results/logs_pretrained_resnet101_multistep_knowledge_mvtec_anomaly_detection_ratio_8_seed_65
    #--resume '/home/cike/workspace/github/fromgithub/mdetr/results/logs_pretrained_EB5_multistep_masked_ratio_8_seed_65_epoch_60_blured/BEST_checkpoint.pth'


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env  --master_port=29504 main.py \
    --dataset_config configs/mvtec_anomaly_detection.json \
    --batch_size 1  \
    --load /home/cike/workspace/data/refer/mdetr/mdetr/checkpoint/refcocog_EB3_checkpoint.pth \
    --ema \
    --text_encoder_lr 1e-5 \
    --text_encoder_type /home/cike/workspace/data/huggingface/roberta-base \
    --lr 5e-5 \
    --lr_drop 10 \
    --schedule multistep \
    --backbone "timm_tf_efficientnet_b3_ns" \
    --output_dir ./results/logs_refcocog_EB3_multistep_knowledge_mvtec_anomaly_detection_ratio_8_seed_65
    #--resume '/home/cike/workspace/github/fromgithub/mdetr/results/logs_pretrained_EB5_multistep_masked_ratio_8_seed_65_epoch_60_blured/BEST_checkpoint.pth'