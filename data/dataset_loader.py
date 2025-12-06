import torch
import functools
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import json
import pickle
from pathlib import Path
base_path = Path(__file__).absolute().parents[1].absolute()
import random
from models.utils.simple_tokenizer import SimpleTokenizer

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


def build_random_masked_tokens_and_labels(tokens, tokenizer):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        # tokenizer = SimpleTokenizer()
        tokens = tokenize(tokens, tokenizer, text_length=77, truncate=True)
        # mask = tokenizer.encoder["<|mask|>"]
        # token_range = list(range(1, len(tokenizer.encoder) - 3))  # 1 ~ 49405
        #
        labels = []
        # for i, token in enumerate(tokens):
        #     if 0 < token < 49405:
        #         prob = random.random()
        #         # mask token with 15% probability
        #         if prob < 0.15:
        #             prob /= 0.15
        #
        #             # 80% randomly change token to mask token
        #             if prob < 0.8:
        #                 tokens[i] = mask
        #
        #             # 10% randomly change token to random token
        #             elif prob < 0.9:
        #                 tokens[i] = random.choice(token_range)
        #
        #             # -> rest 10% randomly keep current token
        #
        #             # append current token to output (we will predict these later)
        #             labels.append(token)
        #         else:
        #             # no masking token (will be ignored by loss function later)
        #             labels.append(0)
        #     else:
        #         labels.append(0)
        #
        # if all(l == 0 for l in labels):
        #     # at least mask 1
        #     labels[1] = tokens[1]
        #     tokens[1] = mask
        # # return torch.tensor(tokens), torch.tensor(labels)
        return tokens, labels


class ImageDatasetClipPRCC(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = SimpleTokenizer()
        with open(base_path / 'data' / 'prcc' / 'caption' / f'cap.train.json') as f:
            self.cap_train = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        if img_path[2:] in self.cap_train[pid]['candidates']:
            target = random.choice(self.cap_train[pid]['targets'])
            cap = random.choice(self.cap_train[pid]['captions'])
            cap_inv = random.choice(self.cap_train[pid]['inv_captions'])
            # target = self.cap_train[pid]['targets'][0]
            # cap = self.cap_train[pid]['captions'][0]
        else:
            target = random.choice(self.cap_train[pid]['targets'])
            cap = random.choice(self.cap_train[pid]['inv_captions'])
            cap_inv = random.choice(self.cap_train[pid]['captions'])
            # target = self.cap_train[pid]['candidates'][0]
            # cap = self.cap_train[pid]['inv_captions'][0]
        img = read_image(img_path)
        target = read_image(target)
        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)
        return img, pid, camid, clothes_id, target, cap, cap_inv


class ImageDatasetClipPRCCTrain(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        # self.tokenizer = SimpleTokenizer()
        # with open(base_path / 'data' / 'prcc' / 'caption' / f'cap.train.json') as f:
        # with open(base_path / 'data' / 'prcc' / 'caption' / f'cap_qwen7B.train.self.json') as f:
        with open(base_path / 'data' / 'captions' / f'prcc.json') as f:
            self.cap_train = json.load(f)
        #
        # with open(base_path / 'data' / 'captions' / f'prcc.pkl', "rb") as f:
        #     self.cap_train = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        caption = self.cap_train[img_path][0]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id, caption

        # img_path, pid, camid, clothes_id = self.dataset[index]
        # if img_path[2:] in self.cap_train[pid]['candidates']:  ##AB
        #     target_path = random.choice(self.cap_train[pid]['targets'])
        #     cap = random.choice(self.cap_train[pid]['captions'])
        #     cap_inv = random.choice(self.cap_train[pid]['inv_captions'])
        #     # target = self.cap_train[pid]['targets'][0]
        #     # cap = self.cap_train[pid]['captions'][0]
        # else: ##C
        #     target_path = random.choice(self.cap_train[pid]['candidates'])
        #     cap = random.choice(self.cap_train[pid]['inv_captions'])
        #     cap_inv = random.choice(self.cap_train[pid]['captions'])
        #     # target = self.cap_train[pid]['candidates'][0]
        #     # cap = self.cap_train[pid]['inv_captions'][0]
        # img = read_image(img_path)
        # target = read_image(target_path)
        # # print(img_path, target_path)
        # if self.transform is not None:
        #     img = self.transform(img)
        #     target = self.transform(target)
        #
        # caption_tokens = tokenize(cap, tokenizer=self.tokenizer, text_length=77, truncate=True)
        # # mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy().copy())
        # # mlm_tokens, mlm_labels =None, None
        # return img, pid, camid, clothes_id, target, cap, cap_inv, img_path, target_path

    # def _build_random_masked_tokens_and_labels(self, tokens):
    #     """
    #     Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    #     :param tokens: list of int, tokenized sentence.
    #     :return: (list of int, list of int), masked tokens and related labels for MLM prediction
    #     """
    #     mask = self.tokenizer.encoder["<|mask|>"]
    #     token_range = list(range(1, len(self.tokenizer.encoder) - 3))  # 1 ~ 49405
    #
    #     labels = []
    #     for i, token in enumerate(tokens):
    #         if 0 < token < 49405:
    #             prob = random.random()
    #             # mask token with 15% probability
    #             if prob < 0.15:
    #                 prob /= 0.15
    #
    #                 # 80% randomly change token to mask token
    #                 if prob < 0.8:
    #                     tokens[i] = mask
    #
    #                 # 10% randomly change token to random token
    #                 elif prob < 0.9:
    #                     tokens[i] = random.choice(token_range)
    #
    #                 # -> rest 10% randomly keep current token
    #
    #                 # append current token to output (we will predict these later)
    #                 labels.append(token)
    #             else:
    #                 # no masking token (will be ignored by loss function later)
    #                 labels.append(0)
    #         else:
    #             labels.append(0)
    #
    #     if all(l == 0 for l in labels):
    #         # at least mask 1
    #         labels[1] = tokens[1]
    #         tokens[1] = mask
    #
    #     return torch.tensor(tokens), torch.tensor(labels)


class ImageDatasetClipLTCC(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        # with open(base_path / 'data' / 'LTCC_ReID' / 'caption' / f'cap.train.json') as f:
        # with open(base_path / 'data' / 'LTCC_ReID' / 'caption' / f'cap_qwen7B_total.train.json') as f:
        # with open(base_path / 'data' / 'LTCC_ReID' / 'caption' / f'cap_qwen7B.train.self.json') as f:
        with open(base_path / 'data' / 'captions' / f'ltcc.json') as f:
            self.cap_train = json.load(f)
        # with open(base_path / 'data' / 'captions' / f'ltcc.pkl', "rb") as f:
        #     self.cap_train = pickle.load(f)
        # self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        # file_path = img_path[5:]
        # print(img_path)
        caption = self.cap_train[img_path][0]
        # caption = tokenize(caption, self.tokenizer, text_length=77, truncate=True)
        # caption, text_labels = build_random_masked_tokens_and_labels(caption, self.tokenizer)
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id, caption

    # def __getitem__(self, index):
    #     img_path, pid, camid, clothes_id = self.dataset[index]
    #     cloth_id = img_path.split('/')[-1].split('_')[1]
    #     list_key = list(self.cap_train[pid].keys())
    #     list_key.remove('id')
    #     list_key.remove(cloth_id)
    #     if len(list_key)==0:
    #         target_key = cloth_id
    #     else:
    #         target_key = random.choice(list_key)
    #     target_path = random.choice(self.cap_train[pid][target_key]['images'])
    #     cap = random.choice(self.cap_train[pid][target_key]['captions'])
    #     cap_inv = random.choice(self.cap_train[pid][cloth_id]['captions'])
    #     img = read_image(img_path)
    #     target = read_image(target_path)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #         target = self.transform(target)
    #     caption_tokens = tokenize(cap, tokenizer=self.tokenizer, text_length=77, truncate=True)
    #     return img, pid, camid, clothes_id, target, cap, cap_inv, img_path, target_path


class ImageDatasetClipVCTrain(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = SimpleTokenizer()
        # with open(base_path / 'data' / 'VC-Clothes' / 'caption' / f'cap.train.json') as f:
        # with open(base_path / 'data' / 'LTCC_ReID' / 'caption' / f'cap_qwen7B_total.train.json') as f:
        with open(base_path / 'data' / 'captions' / f'vcclothes.json') as f:
            self.cap_train = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        # file_path = img_path[5:]
        # print(img_path)
        caption = self.cap_train[img_path][0]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id, caption

    # def __getitem__(self, index):
    #     img_path, pid, camid, clothes_id = self.dataset[index]
    #     cloth_id = img_path.split('/')[-1].split('-')[2]
    #     list_key = list(self.cap_train[pid].keys())
    #     list_key.remove('id')
    #     list_key.remove(cloth_id)
    #     if len(list_key)==0:
    #         target_key = cloth_id
    #     else:
    #         target_key = random.choice(list_key)
    #     target_path = random.choice(self.cap_train[pid][target_key]['images'])
    #     cap = random.choice(self.cap_train[pid][target_key]['captions'])
    #     cap_inv = random.choice(self.cap_train[pid][cloth_id]['captions'])
    #     img = read_image(img_path)
    #     target = read_image(target_path)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #         target = self.transform(target)
    #     caption_tokens = tokenize(cap, tokenizer=self.tokenizer, text_length=77, truncate=True)
    #     return img, pid, camid, clothes_id, target, cap, cap_inv, img_path, target_path


class ImageDatasetLastTrain(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = SimpleTokenizer()
        # with open(base_path / 'data' / 'last' / 'caption' / f'cap.train.json') as f:
        # with open(base_path / 'data' / 'LTCC_ReID' / 'caption' / f'cap_qwen7B_total.train.json') as f:
        with open(base_path / 'data' / 'captions' / f'last.json') as f:
            self.cap_train = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        # file_path = img_path[5:]
        # print(img_path)
        caption = self.cap_train[img_path][0]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id, caption

    # def __getitem__(self, index):
    #     img_path, pid, camid, clothes_id = self.dataset[index]
    #     cloth_id = img_path.split('/')[-1].split('_')[-1].split('.')[0]
    #     list_key = list(self.cap_train[pid].keys())
    #     list_key.remove('id')
    #     list_key.remove(cloth_id)
    #     if len(list_key)==0:
    #         target_key = cloth_id
    #     else:
    #         target_key = random.choice(list_key)
    #     target_path = random.choice(self.cap_train[pid][target_key]['images'])
    #     cap = random.choice(self.cap_train[pid][target_key]['captions'])
    #     cap_inv = random.choice(self.cap_train[pid][cloth_id]['captions'])
    #     img = read_image(img_path)
    #     target = read_image(target_path)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #         target = self.transform(target)
    #     caption_tokens = tokenize(cap, tokenizer=self.tokenizer, text_length=77, truncate=True)
    #     return img, pid, camid, clothes_id, target, cap, cap_inv, img_path, target_path



class ImageDatasetClipTestPRCC(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id, img_path


class ImageDatasetClipLTCCTest(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        with open(base_path / 'data' / 'LTCC_ReID' / 'caption' / f'cap.test.json') as f:
        # with open(base_path / 'data' / 'LTCC_ReID' / 'caption' / f'cap_qwen7B_total.test.json') as f:
            self.cap_test = json.load(f)
        self.id_to_idx={}
        for i, dic in enumerate(self.cap_test):
            self.id_to_idx[int(dic['id'])] = i
        # print(self.id_to_idx)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        idx = self.id_to_idx[pid]
        # cap = self.cap_test[idx]['captions'][0]
        # dic = self.cap_test[idx]['captions']
        cloth_id = img_path.split('/')[-1].split('_')[1]
        list_key = list(self.cap_test[idx].keys())
        list_key.remove('id')
        if len(list_key)>1: list_key.remove(cloth_id)
        # print(list_key)
        # input()
        list_cap = []
        for x in list_key:
            list_cap.extend(self.cap_test[idx][x]['captions'])
        list_query_cap = self.cap_test[idx][cloth_id]['captions']
        cap_query = random.choices(list_query_cap, k=1)
        cap = random.choices(list_cap, k=1)
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id, cap_query

# class ImageDatasetClipLTCCTest(Dataset):
#     """Image Person ReID Dataset"""
#     def __init__(self, dataset, transform=None):
#         self.dataset = dataset
#         self.transform = transform
#         with open(base_path / 'cir_reid' / 'gpt_cap' / f'LTCC_caption.json') as f:
#         # with open(base_path / 'data' / 'LTCC_ReID' / 'caption' / f'cap_qwen7B_total.test.json') as f:
#             self.cap_test = json.load(f)
#         # print(self.id_to_idx)
#     def __len__(self):
#         return len(self.dataset)
#     def __getitem__(self, index):
#         img_path, pid, camid, clothes_id = self.dataset[index]
#         print(img_path)
#         input()
#
#         img = read_image(img_path)
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, pid, camid, clothes_id, cap_query


class ImageDatasetClipVCTest(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        with open(base_path / 'data' / 'VC-Clothes' / 'caption' / f'cap.gallery.json') as f:
        # with open(base_path / 'data' / 'LTCC_ReID' / 'caption' / f'cap_qwen7B_total.test.json') as f:
            self.cap_test = json.load(f)
        self.id_to_idx={}
        for i, dic in enumerate(self.cap_test):
            self.id_to_idx[int(dic['id'])] = i
        # print(self.id_to_idx)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        idx = self.id_to_idx[pid]
        # cap = self.cap_test[idx]['captions'][0]
        # dic = self.cap_test[idx]['captions']
        cloth_id = img_path.split('/')[-1].split('-')[2]
        list_key = list(self.cap_test[idx].keys())
        list_key.remove('id')
        if len(list_key)>1: list_key.remove(cloth_id)

        # print(list_key)
        # input()
        list_cap = []
        for x in list_key:
            list_cap.extend(self.cap_test[idx][x]['captions'])

        cap = random.choices(list_cap, k=1)
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id, cap


class ImageDatasetLastTest(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        with open(base_path / 'data' / 'last' / 'caption' / f'cap.test.json') as f:
        # with open(base_path / 'data' / 'LTCC_ReID' / 'caption' / f'cap_qwen7B_total.test.json') as f:
            self.cap_test = json.load(f)
        self.id_to_idx={}
        for i, dic in enumerate(self.cap_test):
            self.id_to_idx[int(dic['id'])] = i
        # print(self.id_to_idx)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        idx = self.id_to_idx[pid]
        # cap = self.cap_test[idx]['captions'][0]
        # dic = self.cap_test[idx]['captions']
        cloth_id = img_path.split('/')[-1].split('_')[-1].split('.')[0]
        list_key = list(self.cap_test[idx].keys())
        list_key.remove('id')
        if len(list_key)>1: list_key.remove(cloth_id)

        # print(list_key)
        # input()
        list_cap = []
        for x in list_key:
            list_cap.extend(self.cap_test[idx][x]['captions'])

        cap = random.choices(list_cap, k=1)
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id, cap


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id, img_path

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if osp.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self, 
                 dataset, 
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 cloth_changing=True):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.cloth_changing = cloth_changing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        if self.cloth_changing:
            img_paths, pid, camid, clothes_id = self.dataset[index]
        else:
            img_paths, pid, camid = self.dataset[index]

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        clip = self.loader(img_paths)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # trans T x C x H x W to C x T x H x W
        # clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        clip = torch.stack(clip, 0)

        if self.cloth_changing:
            return clip, pid, camid, clothes_id, ''
        else:
            return clip, pid, camid, ''


class VideoDatasetCLIPTrain(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self,
                 dataset,
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 cloth_changing=True):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.cloth_changing = cloth_changing

        with open(base_path / 'data' / 'CCVID' / 'caption' / f'cap.train.json') as f:
        # with open(base_path / 'data' / 'LTCC_ReID' / 'caption' / f'cap_qwen7B_total.train.json') as f:
            self.cap_train = json.load(f)
        self.id_to_idx = {}
        for i, dic in enumerate(self.cap_train):
            self.id_to_idx[int(dic['id'])] = i

        self.change_list = {}
        with open('data/CCVID/{}.txt'.format('train'), 'r', encoding='utf-8') as f:
            for ann in f.readlines():
                ann = ann.strip('\n')  # 去除文本中的换行符
                video_path, person_identity, clothes_label = ann.split('\t')
                self.change_list[video_path] = (person_identity, clothes_label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.cloth_changing:
            img_paths, pid, camid, clothes_id = self.dataset[index]
        else:
            img_paths, pid, camid = self.dataset[index]

        #############
        idx = pid
        vid = img_paths[0].split('/')[3] + '/' + img_paths[0].split('/')[4]
        cloth_id = self.change_list[vid][1]
        list_key = list(self.cap_train[idx].keys())
        list_key.remove('id')
        list_key.remove(cloth_id)
        if len(list_key) == 0:
            target_key = cloth_id
        else:
            target_key = random.choice(list_key)
        target_paths = random.choices(self.cap_train[idx][target_key]['images'], k=8)
        cap = random.choice(self.cap_train[idx][target_key]['captions'])
        cap_inv = random.choice(self.cap_train[idx][cloth_id]['captions'])

        #############

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)
            target_paths = self.temporal_transform(target_paths)

        clip = self.loader(img_paths)
        target_clip = self.loader(target_paths)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            target_clip = [self.spatial_transform(img) for img in target_clip]

        # # trans T x C x H x W to C x T x H x W
        # clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        # trans to T x C x H x W
        clip = torch.stack(clip, 0)
        target_clip = torch.stack(target_clip, 0)

        if self.cloth_changing:
            return clip, pid, camid, clothes_id, target_clip, cap, cap_inv, img_paths, target_paths
        else:
            return clip, pid, camid, target_clip, cap, cap_inv, img_paths, target_paths


        # img_path, pid, camid, clothes_id = self.dataset[index]
        # cloth_id = img_path.split('/')[-1].split('_')[1]
        # list_key = list(self.cap_train[pid].keys())
        # list_key.remove('id')
        # list_key.remove(cloth_id)
        # if len(list_key)==0:
        #     target_key = cloth_id
        # else:
        #     target_key = random.choice(list_key)
        # target_path = random.choice(self.cap_train[pid][target_key]['images'])
        # cap = random.choice(self.cap_train[pid][target_key]['captions'])
        # cap_inv = random.choice(self.cap_train[pid][cloth_id]['captions'])
        # img = read_image(img_path)
        # target = read_image(target_path)
        # if self.transform is not None:
        #     img = self.transform(img)
        #     target = self.transform(target)
        # caption_tokens = tokenize(cap, tokenizer=self.tokenizer, text_length=77, truncate=True)
        # return img, pid, camid, clothes_id, target, cap, cap_inv, img_path, target_path

class VideoDatasetCLIPTest(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self,
                 dataset,
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 cloth_changing=True):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.cloth_changing = cloth_changing

        with open(base_path / 'data' / 'CCVID' / 'caption' / f'cap.gallery.json') as f:
            # with open(base_path / 'data' / 'LTCC_ReID' / 'caption' / f'cap_qwen7B_total.train.json') as f:
            self.cap_test = json.load(f)
        self.id_to_idx = {}
        for i, dic in enumerate(self.cap_test):
            self.id_to_idx[int(dic['id'])] = i

        self.change_list = {}
        with open('data/CCVID/{}.txt'.format('query'), 'r', encoding='utf-8') as f:
            for ann in f.readlines():
                ann = ann.strip('\n')  # 去除文本中的换行符
                video_path, person_identity, clothes_label = ann.split('\t')
                # dicdic = (video_path, person_identity, clothes_label)
                self.change_list[video_path] = (person_identity, clothes_label)

    def __len__(self):
        return len(self.dataset)

    # def __getitem__(self, index):
    #     img_path, pid, camid, clothes_id = self.dataset[index]
    #     idx = self.id_to_idx[pid]
    #     # cap = self.cap_test[idx]['captions'][0]
    #     # dic = self.cap_test[idx]['captions']
    #     cloth_id = img_path.split('/')[-1].split('_')[-1].split('.')[0]
    #     list_key = list(self.cap_test[idx].keys())
    #     list_key.remove('id')
    #     if len(list_key)>1: list_key.remove(cloth_id)
    #
    #     # print(list_key)
    #     # input()
    #     list_cap = []
    #     for x in list_key:
    #         list_cap.extend(self.cap_test[idx][x]['captions'])
    #
    #     cap = random.choices(list_cap, k=1)
    #     img = read_image(img_path)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     return img, pid, camid, clothes_id, cap

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        if self.cloth_changing:
            img_paths, pid, camid, clothes_id = self.dataset[index]
        else:
            img_paths, pid, camid = self.dataset[index]

        #############
        idx = self.id_to_idx[pid]
        vid = img_paths[0].split('/')[3] + '/' + img_paths[0].split('/')[4]
        cloth_id = self.change_list[vid][1]
        list_key = list(self.cap_test[idx].keys())
        list_key.remove('id')
        if len(list_key)>1 and cloth_id in list_key: list_key.remove(cloth_id)
        list_cap = []
        for x in list_key:
            list_cap.extend(self.cap_test[idx][x]['captions'])
        cap = random.choices(list_cap, k=1)

        #############

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        clip = self.loader(img_paths)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # # trans T x C x H x W to C x T x H x W
        # clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        # trans to C x T x H x W
        clip = torch.stack(clip, 0)

        if self.cloth_changing:
            return clip, pid, camid, clothes_id, cap
        else:
            return clip, pid, camid, cap