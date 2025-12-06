import copy
import os
import time
import datetime
import logging
import torch
from torch import optim, nn
from tools.utils import AverageMeter
import numpy as np
import torch.nn.functional as F
from torch import distributed as dist
from tqdm import tqdm
import random
from models.utils.simple_tokenizer import SimpleTokenizer
from typing import Any, Union, List
from pkg_resources import packaging
from losses.gather import GatherLayer
from losses.orthogonal_loss import OrthogonalProjectionLoss
from losses.triplet_loss import TripletLoss


def tokenize(texts: Union[str, List[str]], tokenizer, context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def train_clip_combiner(config, epoch, clip_model, criterion_cla, criterion_pair, optimizer, trainloader, pid2clothes):
    logger = logging.getLogger('cir_reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_opl_loss = AverageMeter()
    corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    crossentropy_criterion = nn.CrossEntropyLoss()
    tokenizer = SimpleTokenizer()
    opl = OrthogonalProjectionLoss()
    triplet_criterion = TripletLoss(margin=0.3)
    scaler = torch.cuda.amp.GradScaler()
    clip_model.train()
    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids, cap) in enumerate(trainloader):
        images_in_batch = imgs.size(0)
        optimizer.zero_grad()
        reference_images = imgs.cuda()
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        text_inputs = tokenize(cap, tokenizer, context_length=77, truncate=True).cuda()
        with torch.cuda.amp.autocast():
            [cls_score, cls_score_proj], [img_feature, img_feature_proj], [com_proj, com_z, t_bn, t_z_bn] = clip_model(reference_images, text_inputs)
            reference_features_proj = img_feature_proj
            cla_loss = criterion_cla(cls_score_proj, pids)
            rn1 = random.gauss(0.5, 0.5)
            cir_com_loss = criterion_pair(reference_features_proj-com_proj*rn1, reference_features_proj-com_proj*rn1, pids)
            opl_loss = opl(reference_features_proj, com_proj, pids, clothes_ids)*0.5
            loss = cla_loss + cir_com_loss + opl_loss

        # Backpropagate and update the weights
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_cla_loss.update(cla_loss.item(), reference_images.size(0))
        batch_pair_loss.update(cir_com_loss.item(), reference_images.size(0))
        batch_opl_loss.update(opl_loss.item(), clothes_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{batch_time.sum:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'ClaLoss:{cla_loss.avg:.4f} '
                'PairLoss:{pair_loss.avg:.4f} '
                'OPLLoss:{mlm_loss.avg:.4f}'.format(
            epoch + 1, batch_time=batch_time, data_time=data_time,
            cla_loss=batch_cla_loss, pair_loss=batch_pair_loss,
            mlm_loss=batch_opl_loss))

