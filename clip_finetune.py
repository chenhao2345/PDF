import os
import sys
import shutil
import time
import datetime
import argparse
import logging
import os.path as osp
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import distributed as dist
import torch.nn.functional as F

from configs.default_img import get_img_config
from configs.default_vid import get_vid_config
from data import build_dataloader
from models import build_model
from losses import build_losses
from tools.utils import save_checkpoint, set_seed, get_logger
from tools.lr_scheduler import WarmupMultiStepLR
from train import train_clip_combiner
from test import test, test_prcc, test_prcc_clip, test_prcc_clip_combiner, test_clip_combiner, concat_all_gather
from models.clip_model import build_CLIP_from_openai_pretrained
import collections

VID_DATASET = ['ccvid']


def element_wise_sum(image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
    """
    Normalized element-wise sum of image features and text features
    :param image_features: non-normalized image features
    :param text_features: non-normalized text features
    :return: normalized element-wise sum of image and text features
    """
    return F.normalize(image_features + text_features, dim=-1)


def parse_option():
    parser = argparse.ArgumentParser(
        description='Train clothes-changing re-id model with clothes-based adversarial loss')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, default='prcc', help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--amp', action='store_true', help="automatic mixed precision")
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')

    args, unparsed = parser.parse_known_args()
    if args.dataset in VID_DATASET:
        config = get_vid_config(args)
    else:
        config = get_img_config(args)

    return config


def main(config, model_folder):
    # Build dataloader
    if config.DATA.DATASET == 'prcc':
        trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler = build_dataloader(
            config)
    else:
        trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(config)
    pid2clothes = torch.from_numpy(dataset.pid2clothes)

    # Build Clip
    # clip_model_name = 'RN50'
    # clip_model_name = 'RN101'
    # clip_model_name = 'ViT-B/32'
    clip_model_name = 'ViT-B/16'
    # clip_model_name = 'ViT-L/14'
    if config.DATA.DATASET == 'prcc':
        num_class = 150
    elif config.DATA.DATASET in ['vcclothes_cc', 'vcclothes_sc', 'vcclothes']:
        num_class = 256
    elif config.DATA.DATASET == 'last':
        num_class = 5000
    else:
        num_class = 77

    print('Loading data from ', config.DATA.DATASET, num_class)
    clip_model, clip_cfg = build_CLIP_from_openai_pretrained(clip_model_name, (config.DATA.HEIGHT, config.DATA.WIDTH), 16, num_class)
    clip_model.eval().float()
    
    # Build identity classification loss, pairwise loss, clothes classificaiton loss, and adversarial loss.
    criterion_cla, criterion_pair, criterion_clothes, criterion_adv = build_losses(config, dataset.num_train_clothes)

    transformer_params = list(map(id, clip_model.transformer.parameters()))
    base_params = filter(lambda p: id(p) not in transformer_params, clip_model.parameters())

    print("Model size: {:.5f}M".format(sum(p.numel() for p in clip_model.visual.parameters())/1000000.0))

    optimizer = optim.Adam([
                               {'params': base_params},
                           ],
                           lr=config.TRAIN.OPTIMIZER.LR,
                           weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)

    scheduler = WarmupMultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE, warmup_factor=0.1,
                                     warmup_iters=10)

    start_epoch = config.TRAIN.START_EPOCH
    if config.MODEL.RESUME:
        logger.info("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        clip_model.load_state_dict(checkpoint['model_state_dict'])

    local_rank = dist.get_rank()
    clip_model = clip_model.cuda(local_rank)

    torch.cuda.set_device(local_rank)
    clip_model = nn.parallel.DistributedDataParallel(clip_model, device_ids=[local_rank], output_device=local_rank)

    if config.EVAL_MODE:
        logger.info("Evaluate only")
        with torch.no_grad():
            if config.DATA.DATASET == 'prcc':
                test_prcc_clip_combiner(clip_model, queryloader_same, queryloader_diff, galleryloader, dataset, combiner)
            else:
                test_clip_combiner(config, clip_model, queryloader, galleryloader, dataset, combiner)
        return

    logger.info("==> Test")
    torch.cuda.empty_cache()

    start_time = time.time()
    if config.DATA.DATASET == 'prcc':
        rank1 = test_prcc_clip_combiner(clip_model, queryloader_same, queryloader_diff, galleryloader, dataset,
                                        query_mode='image')
        total_images = len(dataset.query_same)
    else:
        rank1 = test_clip_combiner(config, clip_model, queryloader, galleryloader, dataset, query_mode='image')
        total_images = len(dataset.query)
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / total_images
    fps = 1.0 / avg_time

    print(f"Total inference time: {total_time:.2f} s")
    print(f"Average time per image: {avg_time:.6f} s")
    print(f"FPS: {fps:.2f}")

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    logger.info("==> Start training")
    for epoch in range(config.TRAIN.MAX_EPOCH):
        train_sampler.set_epoch(epoch)
        start_train_time = time.time()
        train_clip_combiner(config, epoch, clip_model, criterion_cla, criterion_pair, optimizer, trainloader, pid2clothes)

        train_time += round(time.time() - start_train_time)

        if (epoch + 1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and \
                (epoch + 1) % config.TEST.EVAL_STEP == 0 or (epoch + 1) == config.TRAIN.MAX_EPOCH:
            logger.info("==> Test")
            torch.cuda.empty_cache()

            start_time = time.time()
            if config.DATA.DATASET == 'prcc':
                rank1 = test_prcc_clip_combiner(clip_model, queryloader_same, queryloader_diff, galleryloader, dataset,
                                                query_mode='image')
                total_images = len(dataset.query_same)
            else:
                rank1 = test_clip_combiner(config, clip_model, queryloader, galleryloader, dataset, query_mode='image')
                total_images = len(dataset.query)
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / total_images
            fps = 1.0 / avg_time

            print(f"Total inference time: {total_time:.2f} s")
            print(f"Average time per image: {avg_time:.6f} s")
            print(f"FPS: {fps:.2f}")

            torch.cuda.empty_cache()
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

                # if local_rank == 0:
                #     # params1 = clip_model.state_dict()
                #     save_checkpoint(
                #         {
                #         'model_state_dict': clip_model.module.state_dict(),
                #         }
                #         , is_best, osp.join(model_folder, 'model.pth'))

        scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    logger.info("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

    logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))


if __name__ == '__main__':
    config = parse_option()
    # Init dist
    dist.init_process_group(backend="nccl", init_method='env://')
    local_rank = dist.get_rank()
    # Set random seed
    set_seed(config.SEED)
    # get logger
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    final_output_dir = osp.join(config.OUTPUT, 'baseline'+time_str)
    os.makedirs(final_output_dir, exist_ok=True)

    if not config.EVAL_MODE:
        output_file = osp.join(final_output_dir, 'log_train.log')
    else:
        output_file = osp.join(final_output_dir, 'log_test.log')

    logger = get_logger(output_file, local_rank, 'cir_reid')
    logger.info("Config:\n-----------------------------------------")
    logger.info(config)
    logger.info("-----------------------------------------")

    src_folder = osp.join(final_output_dir, 'src')
    model_folder = osp.join(final_output_dir, 'weights')
    if local_rank == 0:
        if not osp.exists(src_folder):
            os.mkdir(src_folder)
            shutil.copytree('models', os.path.join(src_folder, 'models'))
            shutil.copytree('tools', os.path.join(src_folder, 'tools'))
            shutil.copytree('configs', os.path.join(src_folder, 'configs'))
            shutil.copy2('train.py', src_folder)
            shutil.copy2('test.py', src_folder)
            shutil.copy2('clip_finetune.py', src_folder)
        else:
            logger.info("=> src files are already existed in: {}".format(src_folder))
        if not osp.exists(model_folder):
            os.mkdir(model_folder)
        else:
            logger.info("=> model folder already exists")

    main(config, model_folder)