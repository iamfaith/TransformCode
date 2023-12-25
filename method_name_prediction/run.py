from __future__ import absolute_import
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import logging
import argparse
import math
import numpy as np
from io import open
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import (AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer)
                        #   BartConfig, BartForConditionalGeneration, BartTokenizer,
                        #   T5Config, T5ForConditionalGeneration, T5Tokenizer,
                        #   PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer
                        #   )
import multiprocessing
import time
from method_name_prediction_eval import calculate_f1_scores
from java_small_data import CloneDataset
import torch.nn as nn
from pathlib import Path
from commode_utils.metrics import SequentialF1Score
from utils import set_seed

cpu_cont = multiprocessing.cpu_count()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_BIN = 'sim_pytorch_model.bin'

alpha = torch.tensor(0.1, requires_grad=True)

# from model import CodeEmbed, Encoder, CTCLLMModel
from model import MethodNamePredictor

from omegaconf import DictConfig, OmegaConf




def ctc_decode(preds, blank_id):
 
  # blank_id: an integer in the vocab range
  # output_list: a list of output sequences

  # get the input length and batch size
  batch_size, input_length = preds.size()

  # initialize the output list
  output_list = []

  # loop over the batch
  for i in range(batch_size):
    # get the output sequence for the i-th example
    output_seq = preds[i, :]
    

    # remove the blank symbols and adjacent repeated symbols
    output_seq = [x for j, x in enumerate(output_seq) if x != blank_id and (j == 0 or x != output_seq[j-1])]

    # append the output sequence to the output list
    output_list.append(output_seq)

  # return the output list
  return output_list

@torch.no_grad()
def evaluate(args, model, eval_data, write_to_pred=False, is_multi_gpu=False, eval_dataloader=None):
    if eval_dataloader is None:
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation  *****")
    # logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    accuracy = 0
    model.eval()
    logits = []
    y_trues = []
    bar = tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating")

    eval_f1 = SequentialF1Score(pad_idx=CloneDataset.tokenizer.pad_token_id, eos_idx=CloneDataset.tokenizer.eos_token_id, ignore_idx=[CloneDataset.tokenizer.unk_token_id, CloneDataset.tokenizer.bos_token_id])

    _preds, _labels = [], []
    for batch in bar:

        # batch = tuple(t.to(args.device) for t in batch)
        batch.move_to_device(args.device)

        if args.is_ctc:
            # all_source_ids, all_transform_ids, all_source_mask, all_transform_mask, all_method_ids, target_lengths, all_source_mask_len, all_transform_mask_len = batch
            # log_probs = model(all_source_ids)
            code_token, contexts_per_label, labels = batch.code_token, batch.contexts_per_label, batch.labels
            log_probs = model(code_token, labels=None, contexts_per_label=contexts_per_label) 
            preds = log_probs.argmax(-1)
            preds= preds.transpose(0, 1)
            
            _preds.append(preds.cpu())
            _labels.append(labels.cpu())

            # preds = torch.argmax(log_probs, dim=-1)
            output_list = ctc_decode(preds, CloneDataset.tokenizer.pad_token_id)
            # print(CloneDataset.tokenizer.decode(output_list[0], skip_special_tokens=True), CloneDataset.tokenizer.decode(all_method_ids[0], skip_special_tokens=True))

            for pred in output_list:
                logits.append(CloneDataset.tokenizer.decode(pred, skip_special_tokens=True))

            for label in labels:
                y_trues.append(CloneDataset.tokenizer.decode(label, skip_special_tokens=True))

            # print('----')
        else:

            code1 = batch[0].to(args.device)
            code2 = batch[1].to(args.device)
            if args.eval_batch_size != code1.shape[0]:
                continue

            with torch.no_grad():
                outputs = model(code1, code2)
                _, preds = torch.max(outputs, 1)

                # similarity = torch.nn.functional.cosine_similarity(vec1, vec2, dim=1, eps=1e-8) 
                # similarity = torch.abs(similarity)
                

                logits.append(preds.cpu().numpy())

                y_trues.append(batch[2].cpu().numpy())



        nb_eval_steps += 1
        if args.is_ctc:
            f1_score, precision, recall = calculate_f1_scores(logits, y_trues)
            bar.set_description("[{}] Eval F1 score {}, Current step {}".format(nb_eval_steps, round(f1_score, 6), nb_eval_steps))
        else:
            tmp_trues = np.array(y_trues).reshape(-1)
            tmp_prediction = np.array(logits).reshape(-1)
            N = tmp_trues.shape[0]
            accuracy = np.mean(tmp_trues == tmp_prediction)
        # accuracy = (tmp_trues == tmp_prediction).sum() / N
        # TP = ((predictions == 1) & (true_values == 1)).sum()
        # FP = ((predictions == 1) & (true_values == 0)).sum()
        # precision = TP / (TP+FP)

            bar.set_description("[{}] Eval Acc {}, Current step {}".format(nb_eval_steps, round(accuracy, 6), nb_eval_steps))

    if args.is_ctc:

        # for logit in logits:
        #     if '<bos>' in logit:
        #         print('--')
        #     if '<eos>' in logit:
        #         print('--')
        #     if '<unk>' in logit:
        #         print('--')
        #     if '<pad>' in logit:
        #         print('--')
        f1, precision, recall = calculate_f1_scores(logits, y_trues)
        acc = f1

    result = {
        # "eval_accuracy": float(acc),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        # "eval_new_f1": metrics.f1_score.item(),
        # "eval_new_precision": metrics.precision.item(),
        # "eval_new_recall": metrics.recall.item()
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    logger.info("  " + "*" * 20)

    # if write_to_pred:
    #     with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
    #         for example, pred in zip(eval_examples, y_preds):
    #             if pred:
    #                 f.write(example.url1 + '\t' + example.url2 + '\t' + '1' + '\n')
    #             else:
    #                 f.write(example.url1 + '\t' + example.url2 + '\t' + '0' + '\n')

    return result
from typing import cast
from aicmder import AttrDict
def main():
    # parser = argparse.ArgumentParser()
    conf = cast(DictConfig, OmegaConf.load(os.path.dirname(__file__) + "/config.yaml"))
    args = conf.Args
    args = AttrDict(args)
    
    args.is_ctc = True
    args.do_test = True

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    set_seed(args)
    
    config = RobertaConfig()
    # print(config)
    config.max_position_embeddings = 514
    config.type_vocab_size = 1


    cloneData = CloneDataset(tokenizer_file_path=conf.Dataset.tokenizer_file_path, code_tokenizer_file_path=conf.Dataset.code_tokenizer_file_path, data_dir=conf.Dataset.data_dir)
    
    model = MethodNamePredictor(config, bos_id=cloneData.tokenizer.bos_token_id,blank_id=cloneData.tokenizer.pad_token_id, vocab_size=cloneData.tokenizer.vocab_size)
    args.device = device
  
    pool = multiprocessing.Pool(cpu_cont)
    if args.load_model_path is not None and os.path.exists(args.load_model_path):
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    logger.info("Finish loading model from %s", args.load_model_path)
    logger.info(f"[{sum(p.numel() for p in model.parameters() if p.requires_grad)}] trainable parameters")

    model.to(device)  

    logger.info("  " + "***** Testing *****")
    logger.info("  Batch size = %d", args.eval_batch_size)

    for criteria in ['best-f1']:

        #  use load_model_path
        file = os.path.join(args.output_dir, 'checkpoint-{}/{}'.format(criteria, MODEL_BIN))
        if os.path.exists(file):
            try:
                logger.info("Reload model from {}".format(file))
                model.load_state_dict(torch.load(file))
            except:
                logger.info("load model fail!")

        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            # multi-gpu training
            model = torch.nn.DataParallel(model)

        ########################################### origin
        test_dataloader, test_data = cloneData.convert_to_dataset(args=args, pool=pool, tag='test', is_sample=False)


        result = evaluate(args, model, test_data, write_to_pred=True, is_multi_gpu=False, eval_dataloader=test_dataloader)
        logger.info("  test_f1=%.4f", result['eval_f1'])
        logger.info("  test_prec=%.4f", result['eval_precision'])
        logger.info("  test_rec=%.4f", result['eval_recall'])
        logger.info("  " + "*" * 20)


if __name__ == "__main__":
    main()
    
    
    