# coding: utf-8
# Name:     do_pretrain
# Author:   dell
# Data:     2021/11/8
"""
transformers-4.12.3
torch-1.5.0
torchvision=0.6.0
"""
import json
import os
import os
import random
import warnings
from argparse import ArgumentParser
from collections import Counter

import numpy as np
import sentencepiece as spm
import torch
import torch
from KbarTextDataset import KbarTextDataset
from transformers import AutoTokenizer
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import TextDataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint


def main():
    data_root = "/INPUT/lwb/k-emebedding/NORMAL/"
    vocab_path = data_root + "gpt_vocab_bin13_H5/" + "vocab.txt"
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_list = [line.strip() for line in f]
    vocab = {}
    for i, token in enumerate(vocab_list):
        vocab[token] = i
    # 保存词表
    with open(data_root + "gpt2_tokenizer_bin13_H5/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    # 训练sentencepiece模型并生成merge.txt
    # 定义sentencepiece的参数
    # spm_params = '--input_format=text --model_prefix=spm --vocab_size=1000 --character_coverage=1.0 --model_type=unigram'
    # # 训练sentencepiece模型
    # print(spm_params + ' --input='+args.pretrain_data_path)
    # spm.SentencePieceTrainer.Train(spm_params + ' --input='+args.pretrain_data_path)
    # # 加载sentencepiece模型
    # sp = spm.SentencePieceProcessor()
    # sp.Load('spm.model')
    #
    # # 生成merge.txt
    # sp.GetPieceSize()
    # with open('/data/users/lwb/k-emebedding/pretrain_data/gpt2_tokenizer_bin9_0508/merges.txt', 'w', encoding='utf-8') as f:
    #     for i in range(sp.GetPieceSize()):
    #         piece = sp.IdToPiece(i)
    #         f.write(piece + '\n')

    # 定义新的语言对应的词表文件路径
    vocab_file = data_root + "gpt2_tokenizer_bin13_H5/vocab.json"
    merges_file = data_root + "gpt2_tokenizer_bin13_H5/merges.txt"

    # 实例化 tokenizer
    tokenizer = GPT2Tokenizer(vocab_file=vocab_file, merges_file=merges_file)
    tokenizer.save_pretrained(data_root + "/gpt2_tokenizer_bin13_H5")

    # 测试 tokenizer
    text = ["<15min>", "0006d", "lllll"]
    encoded_input = tokenizer.encode(text)
    print(encoded_input)

    # print("=========loading TextDateset=========")
    # dataset = KbarTextDataset(tokenizer=tokenizer, block_size=args.max_seq_len, file_path=args.pretrain_data_path)
    # print("=========TextDateset loaded =========")


if __name__ == "__main__":
    main()
