export CUDA_VISIBLE_DEVICES=9
python -m torch.distributed.run --nproc_per_node=1 --master_port 12352 clip_finetune.py --dataset prcc --cfg configs/prcc.yaml #
# python -m torch.distributed.run --nproc_per_node=1 --master_port 12333 clip_finetune.py --dataset ltcc --cfg configs/ltcc.yaml #