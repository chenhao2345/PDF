import time
import datetime
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
from tools.eval_metrics import evaluate, evaluate_with_clothes
# import clip
import time
from models.utils.simple_tokenizer import SimpleTokenizer
from typing import Any, Union, List
from pkg_resources import packaging

VID_DATASET = ['ccvid']
tokenizer = SimpleTokenizer()
# def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
#     sot_token = tokenizer.encoder["<|startoftext|>"]
#     eot_token = tokenizer.encoder["<|endoftext|>"]
#     tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]
#
#     result = torch.zeros(text_length, dtype=torch.long)
#     if len(tokens) > text_length:
#         if truncate:
#             tokens = tokens[:text_length]
#             tokens[-1] = eot_token
#         else:
#             raise RuntimeError(
#                 f"Input {caption} is too long for context length {text_length}"
#             )
#     result[:len(tokens)] = torch.tensor(tokens)
#     return result

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


def concat_all_gather(tensors, num_total_examples):
    '''
    Performs all_gather operation on the provided tensor list.
    '''
    outputs = []
    for tensor in tensors:
        tensor = tensor.cuda()
        tensors_gather = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0).cpu()
        # truncate the dummy elements added by DistributedInferenceSampler
        outputs.append(output[:num_total_examples])
    return outputs


@torch.no_grad()
def extract_img_feature(model, dataloader):
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids,_) in enumerate(dataloader):
        flip_imgs = torch.flip(imgs, [3])
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        batch_features = model(imgs)
        batch_features_flip = model(flip_imgs)
        batch_features += batch_features_flip
        batch_features = F.normalize(batch_features, p=2, dim=1)

        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
    features = torch.cat(features, 0)

    return features, pids, camids, clothes_ids


def element_wise_sum(image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
    """
    Normalized element-wise sum of image features and text features
    :param image_features: non-normalized image features
    :param text_features: non-normalized text features
    :return: normalized element-wise sum of image and text features
    """
    return F.normalize(image_features + text_features, dim=-1)

@torch.no_grad()
def extract_img_feature_clip(clip_model, dataloader, mode='image', combining_function=element_wise_sum):
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids, cap) in enumerate(dataloader):
        if mode=='image':
            ## image
            flip_imgs = torch.flip(imgs, [3])
            imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
            batch_features = clip_model.module.encode_image(imgs)
            batch_features_flip = clip_model.module.encode_image(flip_imgs)
            batch_features += batch_features_flip
            batch_features = F.normalize(batch_features, p=2, dim=1)

        elif mode=='text':
            ## text
            captions = cap
            text_inputs = tokenize(captions, tokenizer, text_length=77, truncate=True).cuda()
            batch_features = clip_model.module.encode_text(text_inputs)
            batch_features = F.normalize(batch_features, p=2, dim=1)

        else:
            ## image
            flip_imgs = torch.flip(imgs, [3])
            imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
            image_features = clip_model.module.encode_image(imgs)
            image_features_flip = clip_model.module.encode_image(flip_imgs)
            image_features += image_features_flip
            ## text
            captions = cap
            text_inputs = tokenize(captions, tokenizer, text_length=77, truncate=True).cuda()
            text_features = clip_model.module.encode_text(text_inputs)
            ## image+text
            batch_features = combining_function(image_features, text_features)

        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
    features = torch.cat(features, 0)

    return features, pids, camids, clothes_ids

@torch.no_grad()
def extract_vid_feature(model, dataloader, vid2clip_index, data_length):
    # In build_dataloader, each original test video is split into a series of equilong clips.
    # During test, we first extact features for all clips
    logger = logging.getLogger('cir_reid.test')
    clip_features, clip_pids, clip_camids, clip_clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (vids, batch_pids, batch_camids, batch_clothes_ids) in enumerate(dataloader):
        if (batch_idx + 1) % 50==0:
            logger.info("{}/{}".format(batch_idx+1, len(dataloader)))
        vids = vids.cuda()
        batch_features = model(vids)
        clip_features.append(batch_features.cpu())
        clip_pids = torch.cat((clip_pids, batch_pids.cpu()), dim=0)
        clip_camids = torch.cat((clip_camids, batch_camids.cpu()), dim=0)
        clip_clothes_ids = torch.cat((clip_clothes_ids, batch_clothes_ids.cpu()), dim=0)
    clip_features = torch.cat(clip_features, 0)

    # Gather samples from different GPUs
    clip_features, clip_pids, clip_camids, clip_clothes_ids = \
        concat_all_gather([clip_features, clip_pids, clip_camids, clip_clothes_ids], data_length)

    # Use the averaged feature of all clips split from a video as the representation of this original full-length video
    features = torch.zeros(len(vid2clip_index), clip_features.size(1)).cuda()
    clip_features = clip_features.cuda()
    pids = torch.zeros(len(vid2clip_index))
    camids = torch.zeros(len(vid2clip_index))
    clothes_ids = torch.zeros(len(vid2clip_index))
    for i, idx in enumerate(vid2clip_index):
        features[i] = clip_features[idx[0] : idx[1], :].mean(0)
        features[i] = F.normalize(features[i], p=2, dim=0)
        pids[i] = clip_pids[idx[0]]
        camids[i] = clip_camids[idx[0]]
        clothes_ids[i] = clip_clothes_ids[idx[0]]
    features = features.cpu()

    return features, pids, camids, clothes_ids

@torch.no_grad()
def extract_vid_feature_clip(model, dataloader, vid2clip_index, data_length, mode='image'):
    # In build_dataloader, each original test video is split into a series of equilong clips.
    # During test, we first extact features for all clips
    logger = logging.getLogger('cir_reid.test')
    clip_features, clip_pids, clip_camids, clip_clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (vids, batch_pids, batch_camids, batch_clothes_ids, cap) in enumerate(dataloader):
        if (batch_idx + 1) % 50==0:
            logger.info("{}/{}".format(batch_idx+1, len(dataloader)))
        vids = vids.cuda()
        #################
        # print(batch_pids, cap[0], len(cap), len(cap[0]))
        if mode=='image':
            batch_features = model.module.encode_vid(vids)
        else:
            cap = cap[0]
            text_inputs = tokenize(cap, tokenizer, context_length=77, truncate=True).cuda()
            # print(vids.shape, text_inputs.shape)
            batch_features = model.module.combine_vid(vids, text_inputs)
        # input()
        # print(vids.shape)
        #################
        # batch_features = model(vids)
        clip_features.append(batch_features.cpu())
        clip_pids = torch.cat((clip_pids, batch_pids.cpu()), dim=0)
        clip_camids = torch.cat((clip_camids, batch_camids.cpu()), dim=0)
        clip_clothes_ids = torch.cat((clip_clothes_ids, batch_clothes_ids.cpu()), dim=0)
    clip_features = torch.cat(clip_features, 0)

    # Gather samples from different GPUs
    clip_features, clip_pids, clip_camids, clip_clothes_ids = \
        concat_all_gather([clip_features, clip_pids, clip_camids, clip_clothes_ids], data_length)

    # Use the averaged feature of all clips split from a video as the representation of this original full-length video
    features = torch.zeros(len(vid2clip_index), clip_features.size(1)).cuda()
    clip_features = clip_features.cuda()
    pids = torch.zeros(len(vid2clip_index))
    camids = torch.zeros(len(vid2clip_index))
    clothes_ids = torch.zeros(len(vid2clip_index))
    for i, idx in enumerate(vid2clip_index):
        features[i] = clip_features[idx[0] : idx[1], :].mean(0)
        features[i] = F.normalize(features[i], p=2, dim=0)
        pids[i] = clip_pids[idx[0]]
        camids[i] = clip_camids[idx[0]]
        clothes_ids[i] = clip_clothes_ids[idx[0]]
    features = features.cpu()

    return features, pids, camids, clothes_ids


def test(config, model, queryloader, galleryloader, dataset):
    logger = logging.getLogger('cir_reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features 
    if config.DATA.DATASET in VID_DATASET:
        qf, q_pids, q_camids, q_clothes_ids = extract_vid_feature(model, queryloader, 
                                                                  dataset.query_vid2clip_index,
                                                                  len(dataset.recombined_query))
        gf, g_pids, g_camids, g_clothes_ids = extract_vid_feature(model, galleryloader, 
                                                                  dataset.gallery_vid2clip_index,
                                                                  len(dataset.recombined_gallery))
    else:
        qf, q_pids, q_camids, q_clothes_ids = extract_img_feature(model, queryloader)
        gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(model, galleryloader)
        # Gather samples from different GPUs
        torch.cuda.empty_cache()
        qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids], len(dataset.query))
        gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    
    logger.info("Extracted features for query set, obtained {} matrix".format(qf.shape))    
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i+1], gf.t())).cpu()
    distmat = distmat.numpy()
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    time_elapsed = time.time() - since
    logger.info('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    since = time.time()
    logger.info("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    logger.info('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if config.DATA.DATASET in ['last', 'deepchange', 'vcclothes_sc', 'vcclothes_cc']: return cmc[0]

    logger.info("Computing CMC and mAP only for the same clothes setting")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes-changing")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0]


def test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset):
    logger = logging.getLogger('cir_reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features for query set
    qsf, qs_pids, qs_camids, qs_clothes_ids = extract_img_feature(model, queryloader_same)
    qdf, qd_pids, qd_camids, qd_clothes_ids = extract_img_feature(model, queryloader_diff)
    # Extract features for gallery set
    gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(model, galleryloader)
    # Gather samples from different GPUs
    torch.cuda.empty_cache()
    qsf, qs_pids, qs_camids, qs_clothes_ids = concat_all_gather([qsf, qs_pids, qs_camids, qs_clothes_ids], len(dataset.query_same))
    qdf, qd_pids, qd_camids, qd_clothes_ids = concat_all_gather([qdf, qd_pids, qd_camids, qd_clothes_ids], len(dataset.query_diff))
    gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    time_elapsed = time.time() - since
    
    logger.info("Extracted features for query set (with same clothes), obtained {} matrix".format(qsf.shape))
    logger.info("Extracted features for query set (with different clothes), obtained {} matrix".format(qdf.shape))
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n, k = qsf.size(0), qdf.size(0), gf.size(0)
    distmat_same = torch.zeros((m, k))
    distmat_diff = torch.zeros((n, k))
    qsf, qdf, gf = qsf.cuda(), qdf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat_same[i] = (- torch.mm(qsf[i:i+1], gf.t())).cpu()
    for i in range(n):
        distmat_diff[i] = (- torch.mm(qdf[i:i+1], gf.t())).cpu()
    distmat_same = distmat_same.numpy()
    distmat_diff = distmat_diff.numpy()
    qs_pids, qs_camids, qs_clothes_ids = qs_pids.numpy(), qs_camids.numpy(), qs_clothes_ids.numpy()
    qd_pids, qd_camids, qd_clothes_ids = qd_pids.numpy(), qd_camids.numpy(), qd_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    logger.info("Computing CMC and mAP for the same clothes setting")
    cmc, mAP = evaluate(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes changing")
    cmc, mAP = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0]


def test_prcc_clip(model, queryloader_same, queryloader_diff, galleryloader, dataset):
    logger = logging.getLogger('cir_reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features for query set
    qsf, qs_pids, qs_camids, qs_clothes_ids = extract_img_feature_clip(model, queryloader_same, mode='image')
    qdf, qd_pids, qd_camids, qd_clothes_ids = extract_img_feature_clip(model, queryloader_diff, mode='both')
    # Extract features for gallery set
    gf, g_pids, g_camids, g_clothes_ids = extract_img_feature_clip(model, galleryloader, mode='image')
    # Gather samples from different GPUs
    torch.cuda.empty_cache()
    qsf, qs_pids, qs_camids, qs_clothes_ids = concat_all_gather([qsf, qs_pids, qs_camids, qs_clothes_ids],
                                                                len(dataset.query_same))
    qdf, qd_pids, qd_camids, qd_clothes_ids = concat_all_gather([qdf, qd_pids, qd_camids, qd_clothes_ids],
                                                                len(dataset.query_diff))
    gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    time_elapsed = time.time() - since

    logger.info("Extracted features for query set (with same clothes), obtained {} matrix".format(qsf.shape))
    logger.info("Extracted features for query set (with different clothes), obtained {} matrix".format(qdf.shape))
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n, k = qsf.size(0), qdf.size(0), gf.size(0)
    distmat_same = torch.zeros((m, k))
    distmat_diff = torch.zeros((n, k))
    qsf, qdf, gf = qsf.cuda(), qdf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat_same[i] = (- torch.mm(qsf[i:i + 1], gf.t())).cpu()
    for i in range(n):
        distmat_diff[i] = (- torch.mm(qdf[i:i + 1], gf.t())).cpu()
    distmat_same = distmat_same.numpy()
    distmat_diff = distmat_diff.numpy()
    qs_pids, qs_camids, qs_clothes_ids = qs_pids.numpy(), qs_camids.numpy(), qs_clothes_ids.numpy()
    qd_pids, qd_camids, qd_clothes_ids = qd_pids.numpy(), qd_camids.numpy(), qd_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    logger.info("Computing CMC and mAP for the same clothes setting")
    cmc, mAP = evaluate(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes changing")
    cmc, mAP = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0]


@torch.no_grad()
def extract_img_feature_clip_combiner(clip_model, dataloader, mode='image'):
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids, cap) in enumerate(dataloader):
        if mode=='image':
            ## image
            flip_imgs = torch.flip(imgs, [3])
            imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
            batch_features = clip_model(imgs)
            batch_features_flip = clip_model(flip_imgs)
            batch_features += batch_features_flip

            # imgs = imgs.cuda()
            # batch_features = clip_model.module.encode_image(imgs)
            batch_features = F.normalize(batch_features, p=2, dim=1)
        elif mode=='text':
            ## text
            captions = cap[0]
            text_inputs = tokenize(captions, tokenizer, context_length=77, truncate=True).cuda()
            batch_features = clip_model.module.encode_text(text_inputs)
            batch_features = F.normalize(batch_features, p=2, dim=1)
        else:
            # print(batch_pids, cap[0])
            # input()
            ## image
            flip_imgs = torch.flip(imgs, [3]).cuda()
            imgs = imgs.cuda()
            captions = cap[0]
            text_inputs = tokenize(captions, tokenizer, context_length=77, truncate=True).cuda()
            batch_features = clip_model(imgs, text_inputs)
            batch_features_flip = clip_model(flip_imgs, text_inputs)
            # image_features += flip_image_features
            # image_features_before_pooling += flip_image_features_before_pooling
            # batch_features = torch.zeros_like(image_features)
            # for captions in cap:
                # text_inputs = tokenize(captions, tokenizer, context_length=77, truncate=True).cuda()
                # batch_features += clip_model.module.combine(image_features_before_pooling, text_inputs) / len(cap)

            # batch_features = clip_model.module.combine(image_features_before_pooling, text_inputs)
            # batch_features_flip = clip_model.module.combine(flip_image_features_before_pooling, text_inputs)
            batch_features += batch_features_flip
            batch_features = F.normalize(batch_features, p=2, dim=1)
        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
    features = torch.cat(features, 0)

    return features, pids, camids, clothes_ids


@torch.no_grad()
def extract_img_feature_clip_combiner_query_gallery(clip_model, dataloader_q, dataloader_g, mode='image', combining_function=None):
    if mode=='both': print('Combine Image-Text for Evaluation')
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    features_q, captions_q = [], []
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids, cap) in enumerate(dataloader_q):
        features_q.extend()
        captions_q.extend(cap)

    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids, cap) in enumerate(dataloader_g):
        if mode=='image':
            ## image
            flip_imgs = torch.flip(imgs, [3])
            imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
            batch_features = clip_model.module.encode_image(imgs)
            batch_features_flip = clip_model.module.encode_image(flip_imgs)
            batch_features += batch_features_flip

            # imgs = imgs.cuda()
            # batch_features = clip_model.module.encode_image(imgs)
            batch_features = F.normalize(batch_features, p=2, dim=1)

        elif mode=='text':
            ## text
            captions = cap[0]
            text_inputs = tokenize(captions, tokenizer, context_length=77, truncate=True).cuda()
            batch_features = clip_model.module.encode_text(text_inputs)
            batch_features = F.normalize(batch_features, p=2, dim=1)
        else:
            ## image
            # flip_imgs = torch.flip(imgs, [3])
            imgs = imgs.cuda()
            # captions = cap[0]
            image_features, image_features_before_pooling = clip_model.module.encode_image_feat(imgs)
            # flip_image_features, flip_image_features_before_pooling = clip_model.module.encode_image_feat(flip_imgs)
            # image_features += flip_image_features
            # image_features_before_pooling += flip_image_features_before_pooling
            batch_features = torch.zeros_like(image_features)
            # for captions in cap:
                # text_inputs = tokenize(captions, tokenizer, context_length=77, truncate=True).cuda()
                # batch_features += clip_model.module.combine(image_features_before_pooling, text_inputs) / len(cap)
            captions = cap[0]
            text_inputs = tokenize(captions, tokenizer, context_length=77, truncate=True).cuda()
            # image_features_before_pooling = (image_features_before_pooling+flip_image_features_before_pooling)/2
            batch_features += clip_model.module.combine_distill(image_features_before_pooling, text_inputs)
            batch_features = F.normalize(batch_features, p=2, dim=1)
        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
    features = torch.cat(features, 0)

    return features, pids, camids, clothes_ids


def test_prcc_clip_combiner(model, queryloader_same, queryloader_diff, galleryloader, dataset, query_mode='image'):
    logger = logging.getLogger('cir_reid.test')
    since = time.time()
    model.eval()
    # combiner.eval()
    local_rank = dist.get_rank()
    # Extract features for query set
    qsf, qs_pids, qs_camids, qs_clothes_ids = extract_img_feature_clip_combiner(model, queryloader_same, mode='image')
    # start_time=time.perf_counter()
    qdf, qd_pids, qd_camids, qd_clothes_ids = extract_img_feature_clip_combiner(model, queryloader_diff, mode=query_mode)
    # end_time = time.perf_counter()
    # print('Time Cost:', end_time-start_time)
    # Extract features for gallery set
    gf, g_pids, g_camids, g_clothes_ids = extract_img_feature_clip_combiner(model, galleryloader, mode='image')
    # Gather samples from different GPU
    torch.cuda.empty_cache()
    qsf, qs_pids, qs_camids, qs_clothes_ids = concat_all_gather([qsf, qs_pids, qs_camids, qs_clothes_ids],
                                                                len(dataset.query_same))
    qdf, qd_pids, qd_camids, qd_clothes_ids = concat_all_gather([qdf, qd_pids, qd_camids, qd_clothes_ids],
                                                                len(dataset.query_diff))
    gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    time_elapsed = time.time() - since

    logger.info("Extracted features for query set (with same clothes), obtained {} matrix".format(qsf.shape))
    logger.info("Extracted features for query set (with different clothes), obtained {} matrix".format(qdf.shape))
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n, k = qsf.size(0), qdf.size(0), gf.size(0)
    distmat_same = torch.zeros((m, k))
    distmat_diff = torch.zeros((n, k))
    qsf, qdf, gf = qsf.cuda(), qdf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat_same[i] = (- torch.mm(qsf[i:i + 1], gf.t())).cpu()
    for i in range(n):
        distmat_diff[i] = (- torch.mm(qdf[i:i + 1], gf.t())).cpu()
    distmat_same = distmat_same.numpy()
    distmat_diff = distmat_diff.numpy()
    qs_pids, qs_camids, qs_clothes_ids = qs_pids.numpy(), qs_camids.numpy(), qs_clothes_ids.numpy()
    qd_pids, qd_camids, qd_clothes_ids = qd_pids.numpy(), qd_camids.numpy(), qd_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    logger.info("Computing CMC and mAP for the same clothes setting")
    cmc, mAP = evaluate(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes changing")
    cmc, mAP = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0]


def test_clip_combiner(config, model, queryloader, galleryloader, dataset, query_mode='image'):
    logger = logging.getLogger('cir_reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features
    if config.DATA.DATASET in VID_DATASET:
        qf, q_pids, q_camids, q_clothes_ids = extract_vid_feature_clip(model, queryloader,
                                                                  dataset.query_vid2clip_index,
                                                                  len(dataset.recombined_query), mode='image')
        gf, g_pids, g_camids, g_clothes_ids = extract_vid_feature_clip(model, galleryloader,
                                                                  dataset.gallery_vid2clip_index,
                                                                  len(dataset.recombined_gallery), mode='image')
    else:
        qf, q_pids, q_camids, q_clothes_ids = extract_img_feature_clip_combiner(model, queryloader, mode=query_mode)
        gf, g_pids, g_camids, g_clothes_ids = extract_img_feature_clip_combiner(model, galleryloader, mode='image')
        # Gather samples from different GPUs
        torch.cuda.empty_cache()
        qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids],
                                                                len(dataset.query))
        gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids],
                                                                len(dataset.gallery))
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since

    logger.info("Extracted features for query set, obtained {} matrix".format(qf.shape))
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m, n))
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i + 1], gf.t())).cpu()
    distmat = distmat.numpy()
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    time_elapsed = time.time() - since
    logger.info('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    since = time.time()
    logger.info("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    logger.info('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if config.DATA.DATASET in ['last', 'deepchange', 'vcclothes_sc', 'vcclothes_cc']: return cmc[0]

    logger.info("Computing CMC and mAP only for the same clothes setting")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids,
                                     mode='SC')
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes-changing")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids,
                                     mode='CC')
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0]



@torch.no_grad()
def extract_img_feature_mlm(clip_model, dataloader, mode='image', combining_function=None):
    if mode=='both': print('Combine Image-Text for Evaluation')
    elif mode == 'image': print('Use Image for Evaluation')
    elif mode == 'text': print('Use Text for Evaluation')
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids, cap) in enumerate(dataloader):
        if mode=='image':
            ## image
            flip_imgs = torch.flip(imgs, [3])
            imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
            batch_features = clip_model.module.encode_image(imgs)
            batch_features_flip = clip_model.module.encode_image(flip_imgs)
            batch_features += batch_features_flip
            batch_features = F.normalize(batch_features, p=2, dim=1)

        elif mode=='text':
            ## text
            # text_features = torch.zeros(imgs.size(0), clip_model.module.embed_dim)
            # for captions in cap:
                # captions = cap
            cap = cap[0]
            text_inputs = tokenize(cap, tokenizer, context_length=77, truncate=True).cuda()
            text_features = clip_model.module.encode_text(text_inputs).float()
            # text_inputs = clip.tokenize(captions).cuda()
            # batch_features = clip_model.module.encode_text(text_inputs)
            batch_features = F.normalize(text_features, p=2, dim=1)

        else:
            # # ## image
            # # flip_imgs = torch.flip(imgs, [3])
            # # imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
            # # image_features = clip_model.module.base_model.encode_image(imgs)
            # # image_features_flip = clip_model.module.base_model.encode_image(flip_imgs)
            # # image_features += image_features_flip
            # # ## text
            # # text_inputs = clip.tokenize(cap).cuda()
            # # text_features = clip_model.module.base_model.encode_text(text_inputs)
            # # ## image+text
            # # batch_features = clip_model.module.cross_former(text_features, image_features, image_features)
            # # batch_features = batch_features[:, 0, :].float()
            #
            # flip_imgs = torch.flip(imgs, [3])
            # imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
            # text_inputs = clip.tokenize(cap).cuda()
            # # image_feats, text_feats = clip_model.module.base_model(imgs, text_inputs)
            # image_feats = clip_model.module.encode_image(imgs)
            # # image_features_flip = clip_model.module.encode_image(flip_imgs)
            # i_feats = image_feats
            # # if len(image_feats.shape)==3:
            # #     i_feats = image_feats[:, 0, :].float()
            # # else:
            # #     i_feats = image_feats.float()
            # # # i_feats = image_feats.float() # for CLIP ResNet visual model
            # t_feats = clip_model.module.encode_text(text_inputs)
            # # t_feats = text_feats[torch.arange(text_feats.shape[0]), text_inputs.argmax(dim=-1)].float()
            # batch_features = clip_model.module.combiner.predict_features(i_feats.half(), t_feats.half()).float()
            # batch_features = F.normalize(batch_features, p=2, dim=1)
            # # text_features = torch.zeros_like(image_features)
            # # for captions in cap:
            # #     # captions = cap
            # #     text_inputs = clip.tokenize(captions).cuda()
            # #     text_features += clip_model.module.encode_text(text_inputs)/len(cap)
            # ## image+text
            # # batch_features = combining_function(image_features, text_features)

            ## image
            flip_imgs = torch.flip(imgs, [3])
            imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
            image_features = clip_model.module.encode_image(imgs)
            image_features_flip = clip_model.module.encode_image(flip_imgs)
            image_features += image_features_flip
            ## text
            text_features = torch.zeros_like(image_features)
            for captions in cap:
                text_inputs = tokenize(captions, tokenizer, context_length=77, truncate=True).cuda()
                text_features += clip_model.module.encode_text(text_inputs) / len(cap)
            ## image+text
            batch_features = clip_model.module.combiner.combine_features(image_features.half(), text_features.half()).float()

        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
    features = torch.cat(features, 0)

    return features, pids, camids, clothes_ids


def test_prcc_clip_mlm(model, queryloader_same, queryloader_diff, galleryloader, dataset):
    logger = logging.getLogger('cir_reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features for query set
    qsf, qs_pids, qs_camids, qs_clothes_ids = extract_img_feature_mlm(model, queryloader_same, mode='image')
    qdf, qd_pids, qd_camids, qd_clothes_ids = extract_img_feature_mlm(model, queryloader_diff, mode='both')
    # Extract features for gallery set
    gf, g_pids, g_camids, g_clothes_ids = extract_img_feature_mlm(model, galleryloader, mode='image')
    # Gather samples from different GPUs
    torch.cuda.empty_cache()
    qsf, qs_pids, qs_camids, qs_clothes_ids = concat_all_gather([qsf, qs_pids, qs_camids, qs_clothes_ids],
                                                                len(dataset.query_same))
    qdf, qd_pids, qd_camids, qd_clothes_ids = concat_all_gather([qdf, qd_pids, qd_camids, qd_clothes_ids],
                                                                len(dataset.query_diff))
    gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    time_elapsed = time.time() - since

    logger.info("Extracted features for query set (with same clothes), obtained {} matrix".format(qsf.shape))
    logger.info("Extracted features for query set (with different clothes), obtained {} matrix".format(qdf.shape))
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n, k = qsf.size(0), qdf.size(0), gf.size(0)
    distmat_same = torch.zeros((m, k))
    distmat_diff = torch.zeros((n, k))
    qsf, qdf, gf = qsf.cuda(), qdf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat_same[i] = (- torch.mm(qsf[i:i + 1], gf.t())).cpu()
    for i in range(n):
        distmat_diff[i] = (- torch.mm(qdf[i:i + 1], gf.t())).cpu()
    distmat_same = distmat_same.numpy()
    distmat_diff = distmat_diff.numpy()
    qs_pids, qs_camids, qs_clothes_ids = qs_pids.numpy(), qs_camids.numpy(), qs_clothes_ids.numpy()
    qd_pids, qd_camids, qd_clothes_ids = qd_pids.numpy(), qd_camids.numpy(), qd_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    logger.info("Computing CMC and mAP for the same clothes setting")
    cmc, mAP = evaluate(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes changing")
    cmc, mAP = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0]


def test_clip_mlm(config, model, queryloader, galleryloader, dataset):
    logger = logging.getLogger('cir_reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features
    if config.DATA.DATASET in VID_DATASET:
        qf, q_pids, q_camids, q_clothes_ids = extract_vid_feature(model, queryloader,
                                                                  dataset.query_vid2clip_index,
                                                                  len(dataset.recombined_query))
        gf, g_pids, g_camids, g_clothes_ids = extract_vid_feature(model, galleryloader,
                                                                  dataset.gallery_vid2clip_index,
                                                                  len(dataset.recombined_gallery))
    else:
        qf, q_pids, q_camids, q_clothes_ids = extract_img_feature_mlm(model, queryloader, mode='both')
        gf, g_pids, g_camids, g_clothes_ids = extract_img_feature_mlm(model, galleryloader, mode='image')
        # Gather samples from different GPUs
        torch.cuda.empty_cache()
        qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids],
                                                                len(dataset.query))
        gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids],
                                                                len(dataset.gallery))
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since

    logger.info("Extracted features for query set, obtained {} matrix".format(qf.shape))
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m, n))
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i + 1], gf.t())).cpu()
    distmat = distmat.numpy()
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    time_elapsed = time.time() - since
    logger.info('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    since = time.time()
    logger.info("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    logger.info('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if config.DATA.DATASET in ['last', 'deepchange', 'vcclothes_sc', 'vcclothes_cc']: return cmc[0]

    logger.info("Computing CMC and mAP only for the same clothes setting")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids,
                                     mode='SC')
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes-changing")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids,
                                     mode='CC')
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0]