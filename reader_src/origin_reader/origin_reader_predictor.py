#!/usr/bin/evn python
# encoding: utf-8
'''
@author: xiaofenglei
@contact: weijunlei01@163.com
@file: train_qa_base.py
@time: 2020/9/7 17:56
@desc:
'''
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# """Run BERT on SQuAD."""
from __future__ import absolute_import, division, print_function
import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open
import re
import torch
import numpy as np
from torch.multiprocessing import Process
import torch.multiprocessing as mp
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Sampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import pickle
import gc
from transformers import ElectraTokenizer, BertTokenizer, RobertaTokenizer, AlbertTokenizer

from origin_read_examples import read_examples
from origin_convert_example2features import convert_examples_to_features, convert_dev_examples_to_features
from origin_reader_helper import write_predictions, evaluate
from lazy_dataloader import LazyLoadTensorDataset
from config_predict import get_config

sys.path.append("../pretrain_model")
from pretrain_model.changed_model_roberta import ElectraForQuestionAnsweringForwardBest, \
    ElectraForQuestionAnsweringMatchAttention, ElectraForQuestionAnsweringCrossAttention, \
    ElectraForQuestionAnsweringBiAttention, ElectraForQuestionAnsweringCrossAttentionOnReader,\
    ElectraForQuestionAnsweringThreeCrossAttention, \
    ElectraForQuestionAnsweringCrossAttentionOnSent, ElectraForQuestionAnsweringForwardBestWithNoise, \
    ElectraForQuestionAnsweringCrossAttentionWithDP, AlbertForQuestionAnsweringCrossAttention, \
    AlbertForQuestionAnsweringForwardBest, ElectraForQuestionAnsweringQANet, BertForQuestionAnsweringQANet, \
    BertForQuestionAnsweringQANetAttentionWeight, AlbertForQuestionAnsweringQANet, \
    ElectraForQuestionAnsweringQANetAttentionWeight, BertForQuestionAnsweringQANetTrueCoAttention, \
    BertForQuestionAnsweringQANetTwoCrossAttention, ElectraForQuestionAnsweringQANetTrueCoAttention, \
    ElectraForQuestionAnsweringTwoCrossAttention, ElectraForQuestionAnsweringTwoFakeCrossAttention, \
    ElectraForQuestionAnsweringQANetDouble, BertForQuestionAnsweringQANetDoubleCan, \
    ElectraForQuestionAnsweringQANetDoubleCan, ElectraForQuestionAnsweringQANetWithSentWeight, \
    DebertaForQuestionAnsweringQANet, ElectraForQuestionAnsweringDivideNet, \
    ElectraForQuestionAnsweringSelfAttention, ElectraForQuestionAnsweringFFN, \
    ElectraForQuestionAnsweringCoAttention, ElectraForQuestionAnsweringQANetWoCro, \
    ElectraForQuestionAnsweringQANetWoLN, AlbertForQuestionAnsweringForwardBestWithMask, \
    AlbertForQuestionAnsweringQANetWithMask
from pretrain_model.optimization import BertAdam, warmup_linear
# 自定义好的模型
model_dict = {
    'ElectraForQuestionAnsweringForwardBest': ElectraForQuestionAnsweringForwardBest,
    'ElectraForQuestionAnsweringMatchAttention': ElectraForQuestionAnsweringMatchAttention,
    'ElectraForQuestionAnsweringCrossAttention': ElectraForQuestionAnsweringCrossAttention,
    'ElectraForQuestionAnsweringBiAttention': ElectraForQuestionAnsweringBiAttention,
    'ElectraForQuestionAnsweringCrossAttentionOnReader': ElectraForQuestionAnsweringCrossAttentionOnReader,
    'ElectraForQuestionAnsweringThreeCrossAttention': ElectraForQuestionAnsweringThreeCrossAttention,
    'ElectraForQuestionAnsweringCrossAttentionOnSent': ElectraForQuestionAnsweringCrossAttentionOnSent,
    'ElectraForQuestionAnsweringForwardBestWithNoise': ElectraForQuestionAnsweringForwardBestWithNoise,
    'ElectraForQuestionAnsweringCrossAttentionWithDP': ElectraForQuestionAnsweringCrossAttentionWithDP,
    'AlbertForQuestionAnsweringCrossAttention': AlbertForQuestionAnsweringCrossAttention,
    'AlbertForQuestionAnsweringForwardBest': AlbertForQuestionAnsweringForwardBest,
    'ElectraForQuestionAnsweringQANet': ElectraForQuestionAnsweringQANet,
    'BertForQuestionAnsweringQANet': BertForQuestionAnsweringQANet,
    'BertForQuestionAnsweringQANetAttentionWeight': BertForQuestionAnsweringQANetAttentionWeight,
    'AlbertForQuestionAnsweringQANet': AlbertForQuestionAnsweringQANet,
    'ElectraForQuestionAnsweringQANetAttentionWeight': ElectraForQuestionAnsweringQANetAttentionWeight,
    'BertForQuestionAnsweringQANetTrueCoAttention': BertForQuestionAnsweringQANetTrueCoAttention,
    'BertForQuestionAnsweringQANetTwoCrossAttention': BertForQuestionAnsweringQANetTwoCrossAttention,
    'ElectraForQuestionAnsweringQANetTrueCoAttention': ElectraForQuestionAnsweringQANetTrueCoAttention,
    'ElectraForQuestionAnsweringTwoCrossAttention': ElectraForQuestionAnsweringTwoCrossAttention,
    'ElectraForQuestionAnsweringTwoFakeCrossAttention': ElectraForQuestionAnsweringTwoFakeCrossAttention,
    'ElectraForQuestionAnsweringQANetDouble': ElectraForQuestionAnsweringQANetDouble,
    'BertForQuestionAnsweringQANetDoubleCan': BertForQuestionAnsweringQANetDoubleCan,
    'ElectraForQuestionAnsweringQANetDoubleCan': ElectraForQuestionAnsweringQANetDoubleCan,
    'ElectraForQuestionAnsweringQANetWithSentWeight': ElectraForQuestionAnsweringQANetWithSentWeight,
    'DebertaForQuestionAnsweringQANet': DebertaForQuestionAnsweringQANet,
    'ElectraForQuestionAnsweringDivideNet': ElectraForQuestionAnsweringDivideNet,
    'ElectraForQuestionAnsweringSelfAttention': ElectraForQuestionAnsweringSelfAttention,
    'ElectraForQuestionAnsweringFFN': ElectraForQuestionAnsweringFFN,
    'ElectraForQuestionAnsweringCoAttention': ElectraForQuestionAnsweringCoAttention,
    'ElectraForQuestionAnsweringQANetWoCro': ElectraForQuestionAnsweringQANetWoCro,
    'ElectraForQuestionAnsweringQANetWoLN': ElectraForQuestionAnsweringQANetWoLN,
    'AlbertForQuestionAnsweringForwardBestWithMask': AlbertForQuestionAnsweringForwardBestWithMask,
    'AlbertForQuestionAnsweringQANetWithMask': AlbertForQuestionAnsweringQANetWithMask,
}
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
logger = None


def logger_config(log_path, log_prefix='lwj', write2console=True):
    """
    日志配置
    :param log_path: 输出的日志路径
    :param log_prefix: 记录中的日志前缀
    :param write2console: 是否输出到命令行
    :return:
    """
    global logger
    logger = logging.getLogger(log_prefix)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if write2console:
        # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)
    # 为logger对象添加句柄
    logger.addHandler(handler)

    return logger


def get_test_data(args,
                  tokenizer,
                  logger=None,
                  cls_token='',
                  sep_token='',
                  unk_token='',
                  pad_token='',
                  ):
    """ 获取验证集数据 """
    cached_dev_example_file = '{}/dev_example_file_{}_{}_{}_{}'.format(args.feature_cache_path,
                                                                       args.bert_model.split('/')[-1],
                                                                       str(args.max_seq_length),
                                                                       str(args.doc_stride),
                                                                       args.feature_suffix)
    if not os.path.exists(args.feature_cache_path):
        os.makedirs(args.feature_cache_path)
    if os.path.exists(cached_dev_example_file):
        with open(cached_dev_example_file, "rb") as reader:
            dev_examples = pickle.load(reader)
    else:
        dev_examples = read_examples(
            input_file=args.test_file,
            supporting_para_file=args.test_supporting_para_file,
            tokenizer=tokenizer,
            is_training=False)
        with open(cached_dev_example_file, "wb") as writer:
            pickle.dump(dev_examples, writer)
    logger.info('dev examples: {}'.format(len(dev_examples)))
    cached_dev_features_file = '{}/dev_feature_file_{}_{}_{}_{}'.format(args.feature_cache_path,
                                                                        args.bert_model.split('/')[-1],
                                                                        str(args.max_seq_length),
                                                                        str(args.doc_stride),
                                                                        args.feature_suffix)
    if os.path.exists(cached_dev_features_file):
        with open(cached_dev_features_file, "rb") as reader:
            dev_features = pickle.load(reader)
    else:
        dev_features = convert_dev_examples_to_features(
            examples=dev_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            is_training=False,
            cls_token=cls_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
        )
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info(" Saving dev features into cached file %s", cached_dev_features_file)
            with open(cached_dev_features_file, "wb") as writer:
                pickle.dump(dev_features, writer)
    logger.info('dev feature_num: {}'.format(len(dev_features)))
    dev_data = LazyLoadTensorDataset(dev_features, is_training=False)
    dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.val_batch_size)
    return dev_examples, dev_dataloader, dev_features


def generate_preds(file_name, filter_file, eval_examples, answer_dict, sp_preds):
    results = {}
    answers = {}
    sp_facts = {}
    title_ids = {}
    data = json.load(open(file_name, 'r', encoding='utf-8'))
    filter = json.load(open(filter_file, 'r', encoding='utf-8'))
    for d in data:
        title_id = []
        context = d['context']
        fil = filter[d['_id']]
        for indcon, con in enumerate(context):
            for i in range(len(con[1])):
                if con[1][i].strip() != '' or indcon not in fil:
                    title_id.append([con[0], i])
        title_ids[d['_id']] = title_id
    for ee in eval_examples:
        answers[ee.qas_id] = answer_dict[ee.qas_id]['text']
        sp_pred = sp_preds[ee.qas_id]
        sp_title_id = []
        title_id = title_ids[ee.qas_id]
        assert len(sp_pred) == len(title_id)
        for spp, ti in zip(sp_pred, title_id):
            if spp > 0.5:
                sp_title_id.append(ti)
        sp_facts[ee.qas_id] = sp_title_id
    results['answer'] = answers
    results['sp'] = sp_facts
    return results


def dev_evaluate(args, model, dev_dataloader, n_gpu, device, dev_features, tokenizer, dev_examples):
    """ 模型验证 """
    model.eval()
    all_results = []
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logit", "end_logit", "sent_logit"])

    with torch.no_grad():
        for d_step, d_batch in enumerate(tqdm(dev_dataloader, desc="Dev Iteration")):
            d_example_indices = d_batch[-1].squeeze()
            if n_gpu == 1:
                d_batch = tuple(
                    t.squeeze(0).to(device) for t in d_batch[:-1])  # multi-gpu does scattering it-self
            else:
                d_batch = d_batch[:-1]
            input_ids, input_mask, segment_ids, sent_mask, content_len, pq_end_pos = d_batch
            if len(input_ids.shape) < 2:
                input_ids = input_ids.unsqueeze(0)
                segment_ids = segment_ids.unsqueeze(0)
                input_mask = input_mask.unsqueeze(0)
                if isinstance(d_example_indices, torch.Tensor) and len(d_example_indices.shape) == 0:
                    d_example_indices = d_example_indices.unsqueeze(0)
                pq_end_pos = pq_end_pos.unsqueeze(0)
                if sent_mask is not None and len(sent_mask.shape) < 2:
                    # start_positions = start_positions.unsqueeze(0)
                    # end_positions = end_positions.unsqueeze(0)
                    sent_mask = sent_mask.unsqueeze(0)
                    # sent_lbs = sent_lbs.unsqueeze(0)
                    # sent_weight = sent_weight.unsqueeze(0)
            dev_start_logits, dev_end_logits, dev_sent_logits = model(input_ids,
                                                                      input_mask,
                                                                      segment_ids,
                                                                      pq_end_pos=pq_end_pos,
                                                                      sent_mask=sent_mask)
            for idx, example_index in enumerate(d_example_indices):
                dev_start_logit = dev_start_logits[idx].detach().cpu().tolist()
                dev_end_logit = dev_end_logits[idx].detach().cpu().tolist()
                dev_sent_logit = dev_sent_logits[idx].detach().cpu().tolist()
                dev_feature = dev_features[example_index.item()]
                unique_id = dev_feature.unique_id
                all_results.append(
                    RawResult(unique_id=unique_id, start_logit=dev_start_logit, end_logit=dev_end_logit,
                              sent_logit=dev_sent_logit))

    _, preds, sp_pred = write_predictions(tokenizer, dev_examples, dev_features, all_results)
    preds = generate_preds(args.test_file, args.test_supporting_para_file, dev_examples, preds, sp_pred)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file = os.path.join(args.output_dir, args.predict_file)
    json.dump(preds, open(output_file, 'w', encoding='utf-8'))


def run_predict(rank=0, world_size=1):
    parser = get_config()
    args = parser.parse_args()
    # 配置日志文件
    if rank == 0 and not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    log_path = os.path.join(args.log_path, 'log_{}_{}_{}_{}_{}_{}.log'.format(args.log_prefix,
                                                                              args.bert_model.split('/')[-1],
                                                                              args.output_dir.split('/')[-1],
                                                                              args.train_batch_size,
                                                                              args.max_seq_length,
                                                                              args.doc_stride))
    logger = logger_config(log_path=log_path, log_prefix='')
    logger.info('-' * 15 + '所有配置' + '-' * 15)
    logger.info("所有参数配置如下：")
    for arg in vars(args):
        logger.info('{}: {}'.format(arg, getattr(args, arg)))
    logger.info('-' * 30)

    # 分布式训练设置
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        logger.info("start train on nccl!")
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    # 配置随机数
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    # 梯度积累设置
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    if rank == 0 and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # 设置分词器和模型
    cls_token = '[CLS]'
    sep_token = '[SEP]'
    pad_token = '[PAD]'
    unk_token = '[UNK]'
    if 'electra' in args.bert_model.lower():
        if not args.no_network:
            tokenizer = ElectraTokenizer.from_pretrained(args.bert_model,
                                                         do_lower_case=args.do_lower_case)
        else:
            tokenizer = ElectraTokenizer.from_pretrained(args.checkpoint_dir,
                                                         do_lower_case=args.do_lower_case)
    elif 'albert' in args.bert_model.lower():
        cls_token = '[CLS]'
        sep_token = '[SEP]'
        pad_token = '<pad>'
        unk_token = '<unk>'
        if not args.no_network:
            tokenizer = AlbertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        else:
            tokenizer = AlbertTokenizer.from_pretrained(args.checkpoint_dir, do_lower_case=args.do_lower_case)

    elif 'roberta' in args.bert_model.lower():
        cls_token = '<s>'
        sep_token = '</s>'
        pad_token = '<pad>'
        unk_token = '<unk>'
        if not args.no_network:
            tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        else:
            tokenizer = RobertaTokenizer.from_pretrained(args.checkpoint_dir, do_lower_case=args.do_lower_case)
    elif 'bert' in args.bert_model.lower():
        if not args.no_network:
            tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        else:
            tokenizer = BertTokenizer.from_pretrained(args.checkpoint_dir, do_lower_case=args.do_lower_case)
    train_examples = None
    num_train_optimization_steps = None
    model = model_dict[args.model_name].from_pretrained(args.checkpoint_dir)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    dev_examples, dev_dataloader, dev_features = get_test_data(args,
                                                               tokenizer=tokenizer,
                                                               logger=logger,
                                                               cls_token=cls_token,
                                                               sep_token=sep_token,
                                                               unk_token=unk_token,
                                                               pad_token=pad_token
                                                               )
    dev_evaluate(args,
                 model,
                 dev_dataloader,
                 n_gpu,
                 device,
                 dev_features,
                 tokenizer,
                 dev_examples)


if __name__ == "__main__":
    use_ddp = False
    if not use_ddp:
        run_predict()
    else:
        world_size = 2
        processes = []
        for rank in range(world_size):
            p = Process(target=run_predict, args=(rank, world_size))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
