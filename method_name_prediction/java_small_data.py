from aicmder import import_parent
import_parent(__file__)
import sys
from transformers import PreTrainedTokenizerFast
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers import normalizers, pre_tokenizers, decoders
from tokenizers.normalizers import NFD, StripAccents
from collections import Counter
from tokenizers.pre_tokenizers import Whitespace, Punctuation, PreTokenizer, ByteLevel
from tokenizers.models import WordLevel, WordPiece
import tokenizers
import multiprocessing
from code_transformation import transformCode, normalizeCode, extractCodePath

from torch.utils.data import TensorDataset, Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import logging
import os
import random
import torch
import time
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
# from _utils import *
# from utils import read_examples, calc_stats
import pickle5 as pickle
import pandas as pd
from itertools import chain
import torch.nn.functional as F
logger = logging.getLogger(__name__)
os.environ['TOKENIZERS_PARALLELISM'] = "0"

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                code,
                transform_code,
                code_len,
                transform_code_len,
                method_name,
                method_ids,
                file_path,
                origin_code=None,
                source=None
                 ):
        self.code = code
        self.transform_code = transform_code
        self.code_len = code_len
        self.transform_code_len = transform_code_len
        self.method_name = method_name
        self.method_ids = method_ids
        self.file_path = file_path
        self.origin_code = origin_code
        self.source = source

def convert_example_to_features(item):
    example, example_index, args = item
    example, codeFile = example[:-1], example[-1]
    ret = []
    max_seq_length = args.max_source_length + args.max_target_length
    for method_name, source, transform, origin_code in example:
        if args.is_ctc:
            # # ########### old
            # code = CloneDataset.code_tokenizer.encode_plus(
            #     source, max_length=args.max_source_length, padding='max_length', truncation=True,
            #     add_special_tokens=False)
            # transformCode = CloneDataset.code_tokenizer.encode_plus(
            #     transform, max_length=args.max_source_length, padding='max_length', truncation=True,
            #     add_special_tokens=False)
            ############ old
            code = CloneDataset.code_tokenizer.encode(source)
            transformCode = CloneDataset.code_tokenizer.encode(transform)

            method_ids = [CloneDataset.tokenizer.bos_token_id]
            method_name.append(CloneDataset.tokenizer.eos_token)
            for name in method_name:
                method_ids.extend(CloneDataset.tokenizer.encode(name))
            # method_ids = CloneDataset.tokenizer.encode(method_name)
            # method_ids.append(CloneDataset.tokenizer.eos_token_id)

            ret.append(InputFeatures(code=code, transform_code=transformCode, code_len=len(code), transform_code_len=len(transformCode), method_name=method_name, method_ids=method_ids, file_path=codeFile, origin_code=origin_code, source=source))
        else:
            code = CloneDataset.code_tokenizer.encode(
                source, max_length=args.max_source_length, truncation=True,
                add_special_tokens=True)
            transformCode = CloneDataset.code_tokenizer.encode(
                transform, max_length=args.max_source_length, truncation=True,
                add_special_tokens=True)


            # print(len(transformCode['input_ids']), len(CloneDataset.code_tokenizer.encode(transform)))
            # method_ids = []
            # for name in method_name:
            #     method_ids.extend()

            # if len(method_name) > 1:
                # print(method_name)

            methods = CloneDataset.tokenizer.encode(' '.join(method_name) + CloneDataset.tokenizer.eos_token, add_special_tokens=False, truncation=True, max_length=args.max_target_length)



            code_ids = code + methods
            pad_len = max_seq_length - len(code_ids)
            code_ids = code_ids + [CloneDataset.tokenizer.pad_token_id] * pad_len


            context_length = len(code)
            labels = [CloneDataset.tokenizer.pad_token_id] * context_length + methods
            labels = labels + [CloneDataset.tokenizer.pad_token_id] * pad_len
            labels = [(l if l != CloneDataset.tokenizer.pad_token_id else -100) for l in labels]
            ret.append(InputFeatures(code=code_ids, transform_code=None, method_name=method_name, method_ids=labels, file_path=codeFile))
            # print("code", len(code_ids), len(labels))

            if transform != code:
                code_ids = transformCode + methods
                pad_len = max_seq_length - len(code_ids)
                code_ids = code_ids + [CloneDataset.tokenizer.pad_token_id] * pad_len


                context_length = len(transformCode)
                labels = [CloneDataset.tokenizer.pad_token_id] * context_length + methods
                labels = labels + [CloneDataset.tokenizer.pad_token_id] * pad_len
                labels = [(l if l != CloneDataset.tokenizer.pad_token_id else -100) for l in labels]

                # print("transform code", len(code_ids), len(labels))
                ret.append(InputFeatures(code=code_ids, transform_code=None, method_name=method_name, method_ids=labels, file_path=codeFile))
    return ret



class CodeContextDataset(Dataset):
    def __init__(self, features):
        self.features = features
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        f = self.features[index]
        max_len = 7
        target_len = len(f.method_ids)
        paddingNum = max_len - target_len
        if paddingNum > 0:
            f.method_ids.extend([0 for _ in range(paddingNum)])
        codes, _temp, max_code_len = [], [], 50
        for code in f.code:
            _temp.append(code)
            if len(_temp) == max_code_len:
                codes.append(_temp)
                _temp = []
        if len(_temp) < max_code_len:
            _temp.extend([0 for _ in range(max_code_len - len(_temp))])
            codes.append(_temp)
        return f.method_ids, target_len, codes, f.transform_code, len(codes), f.transform_code_len

from typing import cast, List
def transpose(list_of_lists: List[List[int]]) -> List[List[int]]:
    return [cast(List[int], it) for it in zip(*list_of_lists)]

class BatchedLabeledPathContext:
    def __init__(self, all_samples):
        samples = [s for s in all_samples if s is not None]

        # [max label parts; batch size]
        self.labels = torch.tensor(transpose([s[0] for s in samples]), dtype=torch.long).transpose(0,1)
       
        self.code_token = torch.tensor(transpose([path for s in samples for path in s[2]]), dtype=torch.long).transpose(0, 1)

        # [path length; n contexts]
        # self.transfor_token = torch.tensor(transpose([path for s in samples for path in s[3]]), dtype=torch.long)

        # [batch size]
        self.contexts_per_label = torch.tensor([s[4] for s in samples], dtype=torch.long)
        # self.transform_contexts_per_label = torch.tensor([s[5] for s in samples], dtype=torch.long)

        # print(all_samples)

    def __len__(self) -> int:
        return len(self.contexts_per_label)

    def __get_all_tensors(self): # -> Iterable[Tuple[str, torch.Tensor]]:
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                yield name, value

    def pin_memory(self):
        for name, value in self.__get_all_tensors():
            setattr(self, name, value.pin_memory())
        return self

    def move_to_device(self, device: torch.device):
        for name, value in self.__get_all_tensors():
            setattr(self, name, value.to(device))


class CloneDataset:

    train_data = None
    val_data = None
    test_data = None

    batch_size = 128
    is_debug = False
    # is_debug = True

    tokenized_texts = []
    code_path = []
    tokenizer = None
    code_tokenizer = None
    # tokenizer_file_path = "./custom_tokenizer/method_name_dict_5000.json"
    # code_tokenizer_file_path = "./custom_tokenizer/WordPiece_tokenizer.json"

    def __init__(self, tokenizer_file_path, code_tokenizer_file_path, data_dir):

        # run name_predict_util.py
        with open(f"{data_dir}/test_name_predict.pkl", 'rb') as file:
        # dump information to that file
            CloneDataset.test_data = self.filerNone(pickle.load(file))
            CloneDataset.test_data = np.array(CloneDataset.test_data)

        with open(f"{data_dir}/training_name_predict.pkl", 'rb') as file:
        # dump information to that file
            CloneDataset.train_data = self.filerNone(pickle.load(file))
            CloneDataset.train_data = np.array(CloneDataset.train_data)

        with open(f"{data_dir}/validation_name_predict.pkl", 'rb') as file:
            CloneDataset.val_data = self.filerNone(pickle.load(file))
            CloneDataset.val_data = np.array(CloneDataset.val_data)


        # debug
        if CloneDataset.is_debug:
            CloneDataset.train_data = CloneDataset.train_data[:CloneDataset.batch_size]
            CloneDataset.val_data = CloneDataset.val_data[:CloneDataset.batch_size]
            CloneDataset.test_data = CloneDataset.test_data[:CloneDataset.batch_size]

        vocab_size = 10_000
        unk_token = '<unk>'
        special_tokens = ["<pad>", "<bos>", "<unk>", "<eos>"]
        # special_tokens = {
        #     "unk_token": '<unk>',
        #     "pad_token": "<pad>",
        #     "mask_token": "<mask>"
        # }

        
        CloneDataset.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file_path)
        CloneDataset.tokenizer.add_special_tokens({'pad_token': '<pad>', 'eos_token': '<eos>', 'unk_token': '<unk>', 'bos_token': '<bos>'})
        print("----", CloneDataset.tokenizer.eos_token, self.tokenizer.vocab_size)
    # SPECIAL_TOKENS_ATTRIBUTES = [
    #     "bos_token",
    #     "eos_token",
    #     "unk_token",
    #     "sep_token",
    #     "pad_token",
    #     "cls_token",
    #     "mask_token",
    #     "additional_special_tokens",
    # ]
        
        CloneDataset.code_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=code_tokenizer_file_path)
        if CloneDataset.code_tokenizer.pad_token is None:
            CloneDataset.code_tokenizer.add_special_tokens({'pad_token': '<pad>', "mask_token": '<mask>'})

                # self.model.resize_token_embeddings(len(self.tokenizer))
                # self.model.config.pad_token_id = CloneDataset.code_tokenizer.pad_token_id

        print("code_tokenizer", CloneDataset.code_tokenizer.pad_token, CloneDataset.code_tokenizer.mask_token)


    def filerNone(self, data, min_path_len=20, max_method_len=5):
        ret = []
        for item in data:
            if len(item) > 0:
                methods = []
                for method in item[:-1]:
                    if len(method[1]) >= min_path_len:
                        if len(method[0]) > max_method_len:
                            continue
                        methods.append(method)
                        CloneDataset.code_path.append(method[1])
                        CloneDataset.code_path.append(method[2])
                        CloneDataset.tokenized_texts.extend(method[0])
                if len(methods) > 0:
                    methods.append(item[-1])
                    ret.append(methods)
        return ret

    def collate_wrapper(self, batch):
        return BatchedLabeledPathContext(batch)

    def convert_to_dataset(
            self, args=None, pool=None, tag='train', is_sample=False,
            sample_ratio=0.1):


        ret_data = None
        if tag == "train":
            ret_data = CloneDataset.train_data
        elif tag == "val":
            ret_data = CloneDataset.val_data
        elif tag == "test":
            ret_data = CloneDataset.test_data
        else:
            raise ValueError("invalid " + tag)
        if is_sample:
            size = ret_data.shape[0]
            ret_data = ret_data[np.random.randint(0, size, (int(size * sample_ratio),))]
        print(ret_data.shape)

        # for idx, example in enumerate(ret_data):
        #     print(idx, example)
        #     break

        tuple_examples = [(example, idx, args) for idx, example in enumerate(ret_data)]

        # debug
        # for item in tuple_examples:
        #     convert_example_to_features(item)
        #     break

        features_list = pool.map(convert_example_to_features, tqdm(
            tuple_examples, total=len(tuple_examples)))
        features = list(chain(*features_list))
        # print('----')

        if not args.is_ctc:
            all_source_ids = torch.tensor([f.code for f in features], dtype=torch.float32)
            all_method_ids = torch.tensor([f.method_ids for f in features], dtype=torch.float32)
            data = TensorDataset(all_source_ids, all_method_ids)
            return data
        else:
            codeDataset = CodeContextDataset(features)
            shuffle = tag == 'train'
            batch_size = args.train_batch_size
            # if args.local_rank == -1:
            #     print("RandomSampler")
            #     train_sampler = RandomSampler(codeDataset)
            # else:
            #     print("DistributedSampler")
            #     train_sampler = DistributedSampler(codeDataset)

            return DataLoader(
                codeDataset,
                batch_size,
                shuffle=shuffle,
                num_workers=16,
                # sampler=train_sampler,
                collate_fn=self.collate_wrapper,
                pin_memory=True,
            ), codeDataset

