export CUDA_VISIBLE_DEVICES=5

python test.py \
    --config configs/VLTVG_R101_flickr_1.py \
    --checkpoint work_dirs/VLTVG_R101_flickr_blurred_batch_2/checkpoint0090.pth \
    --batch_size_test 1 \
    --defined_split blurred \
    --test_split val