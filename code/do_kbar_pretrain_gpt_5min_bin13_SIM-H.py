# coding: utf-8
# Name:     do_pretrain
# Author:   dell
# Data:     2021/11/8
"""
transformers-4.12.3
torch-1.5.0
torchvision=0.6.0
"""
import torch.nn as nn
import json
import os
import random
import string
import warnings
from argparse import ArgumentParser
from collections import Counter

import numpy as np
import torch
from KbarTextDataset import KbarTextDataset
from gpt2_models import GPT2LMSIMHeadModel, GPT2LMSimClsHeadModel
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import TextDataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# torch.cuda.is_available()

def load_file(filename):
    with open(filename, 'r', encoding='gb2312') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def count_words(lines):
    words = []
    for line in lines:
        line = line.translate(str.maketrans('', '', string.punctuation))  # 去除标点符号
        words += line.lower().split()
    return Counter(words)


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)  # 为cpu分配随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为gpu分配随机种子
        torch.cuda.manual_seed_all(seed)  # 若使用多块gpu，使用该命令设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmard = False


def main():
    data_root = "/INPUT/lwb/QUANT/data"
    model_root = "/INPUT/lwb/QUANT/model"
    parser = ArgumentParser()
    # model
    parser.add_argument("--pretrain_data_path", type=str,
                        default=data_root + "/hybrid_data_train/5min/5min_lseq_total_FUT_filtered_v2.txt")
    parser.add_argument("--checkpoint_save_path", type=str,
                        default=model_root + "/checkpoint_kbar_bin13_SIM-CLS_H5_total-lseq-contract")
    parser.add_argument("--save_path", type=str, default=model_root + "/H5_total_lseq-gpt-base-kbar-bin13-SIM-CLS-contract")
    # token
    parser.add_argument("--pretrain_cls_labels_path", type=str,
                        default=data_root + "/gpt2_tokenizer_bin13_H5/cls_labels.json")
    parser.add_argument("--pretrain_token_vecs_dict_path", type=str,
                        default=data_root + "/gpt2_tokenizer_bin13_H5/token_vecs_dict.json")
    parser.add_argument("--pretrain_model_path", type=str, default=data_root + "/ckpt/===")

    parser.add_argument("--config_path", type=str,
                        default=data_root + "/gpt_vocab_bin13_H5/config.json")
    parser.add_argument("--tokenizer_path", type=str,
                        default=data_root + "/gpt2_tokenizer_bin13_H5")
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=20)  # 限制checkpoints的数量，最多15个

    # python通过调用warnings模块中定义的warn()函数来发出警告，我们可以通过警告过滤器进行控制是否发出警告消息。
    warnings.filterwarnings("ignore")
    args = parser.parse_args()

    setup_seed(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.do_basic_tokenize = False

    # weights setting
    filename = args.pretrain_data_path
    sentences = load_file(filename)
    word_counts = count_words(sentences)
    total_words = sum(word_counts.values())
    word_weights_dict = {word: (total_words / count) ** 0.5 if "<" not in word else 0 for word, count in word_counts.items()}
    word_weights_values = np.array(list(word_weights_dict.values()))
    word_weights_values = (word_weights_values - word_weights_values.min()) / word_weights_values.max()
    word_weights_dict = dict(zip(word_weights_dict.keys(), word_weights_values))
    # print(word_weights_dict)
    # 定义词汇表
    vocab = tokenizer.get_vocab()
    # word_weights = torch.zeros(161056)
    word_weights = torch.zeros(len(vocab.keys()))
    for word in vocab:
        if word in word_weights_dict:
            word_weights[tokenizer.convert_tokens_to_ids(word)] = word_weights_dict[word]
    print(word_weights.shape, len(vocab))

    # 定义CLS的字典
    cls_labels = []
    with open(args.pretrain_cls_labels_path, "r", encoding='gb2312') as f:
        cls_labels_dict = json.load(f)
    for word in vocab:
        cls_labels.append(cls_labels_dict[word])
    cls_labels = torch.tensor(cls_labels).unsqueeze(dim=1)


    # 定义token_vec的字典
    token_embed = []
    with open(args.pretrain_token_vecs_dict_path, "r", encoding='gb2312') as f:
        token_vecs_dict = json.load(f)
    for word in vocab:
        token_embed.append(token_vecs_dict[word])
    token_vecs = torch.tensor(token_embed)
    # print(token_embed.shape, len(vocab))
    # token_embed = nn.Embedding(len(vocab), 5, _weight=token_embed)

    training_corpus = []
    with open(args.pretrain_data_path, 'r', encoding='gb2312') as file:
        for line in file:
            training_corpus.append(line.strip().split(' '))
    print("sentence lines:", len(training_corpus))

    gpt2_config = GPT2Config.from_pretrained(args.config_path)
    # GPT2LMSimClsHeadModel
    model = GPT2LMSimClsHeadModel(config=gpt2_config, cls_labels=cls_labels, weight=word_weights, token_vecs=token_vecs)
    # model = GPT2LMSIMHeadModel(config=gpt2_config, cls_labels=cls_labels, weight=word_weights, token_vecs=token_vecs)
    model = model.to(device)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, mlm_probability=0.15)

    training_args = TrainingArguments(
        seed=args.seed,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        output_dir=args.checkpoint_save_path,
        learning_rate=args.learning_rate,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size
    )

    print("=========loading TextDateset=========")
    dataset = KbarTextDataset(tokenizer=tokenizer, block_size=args.max_seq_len, file_path=args.pretrain_data_path)
    print("=========TextDateset loaded =========")

    trainer = Trainer(model, args=training_args, train_dataset=dataset, data_collator=data_collator)
    # if torch.cuda.is_available():
    #     model.weight = model.weight.to(trainer.args.device)
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("=========training=========")
        train_result = trainer.train()
    print(train_result)
    trainer.save_model(args.save_path)
    tokenizer.save_vocabulary(args.save_path)


if __name__ == "__main__":
    main()
