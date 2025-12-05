export CUDA_VISIBLE_DEVICES=8
# python -m torch.distributed.run --nproc_per_node=1 --master_port 12242 clip_finetune.py --dataset prcc --cfg configs/prcc.yaml #
python -m torch.distributed.run --nproc_per_node=1 --master_port 12336 clip_finetune.py --dataset ltcc --cfg configs/ltcc.yaml #
#python -m torch.distributed.run --nproc_per_node=1 --master_port 12346 clip_finetune.py --dataset vcclothes --cfg configs/vcclothes.yaml #
# python -m torch.distributed.run --nproc_per_node=1 --master_port 12339 clip_finetune.py --dataset last --cfg configs/last.yaml #