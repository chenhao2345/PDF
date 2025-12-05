#export CUDA_VISIBLE_DEVICES=6,7,8,9
export CUDA_VISIBLE_DEVICES=8
python -m torch.distributed.run --nproc_per_node=1 --master_port 12337 final_test.py --dataset prcc --cfg configs/prcc_test.yaml

#python -m torch.distributed.run --nproc_per_node=1 --master_port 12336 final_test.py --dataset ltcc --cfg configs/ltcc_test.yaml


#python -m torch.distributed.run --nproc_per_node=1 --master_port 12346 clip_finetune.py --dataset vcclothes_sc --cfg configs/vcclothes.yaml  --eval --resume logs/vcclothes/res50-clip-combiner-vcclothes/baseline2025-06-10-10-02-27/weights/best_model.pth #
