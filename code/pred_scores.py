import json
import os
import string
from argparse import ArgumentParser
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import scipy.stats as stats
# from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F
import torchmetrics
from rich.progress import track
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers import GPT2ForSequenceClassification, GPT2LMHeadModel, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers.models.gpt2.tokenization_gpt2_kbar import GPT2TokenizerKbar

# 加载预训练模型和分词器
item_name = 'A'
tick_name = '5min'

def load_file(filename):
    with open(filename, 'r', encoding="gb2312") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def count_words(lines):
    words = []
    for line in lines:
        line = line.translate(str.maketrans('', '', string.punctuation))  # 去除标点符号
        words += line.lower().split()
    return Counter(words)

parser = ArgumentParser()
parser.add_argument("--pretrain_token_vecs_dict_path", type=str,
                    default="/INPUT/lwb/k-emebedding/NORMAL/5minData_bin9/preprocessed_data/token_vecs_dict.json")
parser.add_argument("--pretrain_cls_labels_path", type=str,
                    default="/INPUT/lwb/k-emebedding/NORMAL/5minData_bin9/preprocessed_data/cls_labels.json")

# parser.add_argument("--save_path", type=str, default="/data/users/lwb/pretrain_data/NORMAL/5min-gpt-base-kbar-bin9-CLS")
# parser.add_argument("--save_path", type=str, default="/INPUT/lwb/k-emebedding/NORMAL/5min-gpt-base-kbar-bin9-SIM")
parser.add_argument("--save_path", type=str, default="/INPUT/lwb/k-emebedding/NORMAL/5min-gpt-base-kbar-bin9-CLS")
# parser.add_argument("--save_path", type=str, default="/INPUT/lwb/k-emebedding/NORMAL/5min-gpt-base-kbar-bin9")
args = parser.parse_args()

use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

# 加载tokenizer
tokenizer = GPT2TokenizerKbar.from_pretrained("/INPUT/lwb/k-emebedding/NORMAL/gpt2_tokenizer_bin9",
                                              encoding='gb2312')
tokenizer.pad_token = tokenizer.eos_token

# 加载预训练模型
model = GPT2LMHeadModel.from_pretrained(args.save_path)
model.config.pad_token_id = model.config.eos_token_id

# 生成文本
# 准备数据集：
st = 15
min_period_list = list(range(st, st + 1))

test_save_path = '/INPUT/lwb/k-emebedding/NORMAL/' + tick_name + 'Data_bin9/datasets_for_items/' + item_name + '/'
total_save_path = '/INPUT/lwb/k-emebedding/NORMAL/' + tick_name + 'Data_bin9/preprocessed_data/'
val_save_path = '/INPUT/lwb/k-emebedding/NORMAL/' + tick_name + 'Data_Val_bin9/datasets_for_items/' + item_name + '/' + "data_val.txt"

# 读取数据并计算每个单词在原始数据集中出现的次数
filename = total_save_path + "5min.txt"
sentences = load_file(filename)
word_counts = count_words(sentences)
total_words = sum(word_counts.values())

word_weights = {word: (total_words / count) ** 0.5 for word, count in word_counts.items()}
word_weights_values = np.array(list(word_weights.values()))
word_weights_values = (word_weights_values - word_weights_values.min()) / word_weights_values.max()
word_weights = dict(zip(word_weights.keys(), word_weights_values))
# 定义词汇表
vocab = tokenizer.get_vocab()
word_weights_list = torch.zeros(161056)
for word in vocab:
    if word in word_weights:
        word_weights_list[tokenizer.convert_tokens_to_ids(word)] = word_weights[word]

f = open(test_save_path + "data.txt", 'r', encoding="gb2312")
test_data = f.readlines()
f.close()

# f = open(val_save_path + "data_val.txt", 'r', encoding="utf-8")
f = open(val_save_path, 'r', encoding="gb2312")
val_data = f.readlines()
f.close()

train_datas = []
train_csv_dict = {}
train_labels = []
val_datas = []
val_csv_dict = {}
val_labels = []

print("prepare training data..")

# prepare text data
for i in track(range(len(test_data[0:100]))):
    kbars = test_data[i].replace('\n', '').split(' ')
    idxs = list(range(max(min_period_list), len(kbars) - 1))
    for idx in idxs:
        train_csv_dict[len(train_datas)] = i
        train_feat = kbars[0:idx]
        train_datas.append(' '.join(train_feat))

for i in track(range(len(val_data[0:]))):
    kbars = val_data[i].replace('\n', '').split(' ')
    idxs = list(range(max(min_period_list), len(kbars) - 1))
    for idx in idxs:
        val_csv_dict[len(val_datas)] = i
        train_feat = kbars[0:idx]
        val_datas.append(' '.join(train_feat))

print("numbers: ", len(val_datas))
input_ids_train = []
labels_train = []
padding_token_id = 0
for sent in train_datas:
    input_ids_train.append(torch.tensor(tokenizer.convert_tokens_to_ids(sent.split(' '))))

input_ids_val = []
labels_val = []
for sent in val_datas:
    input_ids_val.append(torch.tensor(tokenizer.convert_tokens_to_ids(sent.split(' '))))

sample_idx = 0
input_text = train_datas[sample_idx][5:]
total_text = train_datas[sample_idx + 1][5:]
input_id_train = input_ids_train[sample_idx].unsqueeze(dim=0)

# 使用模型进行预测
with torch.no_grad():
    outputs = model(input_id_train)
    predictions = outputs[0]

# 获取下一个单词的预测概率分布
next_token_predictions = predictions[0, -1, :]  #
top_k = 10
print(next_token_predictions, len(next_token_predictions), max(next_token_predictions))
# 获取前k个最有可能的单词及其对应的概率
top_k_tokens = torch.topk(next_token_predictions, top_k, dim=-1).indices.tolist()
top_k_probabilities = torch.softmax(torch.topk(next_token_predictions, top_k, dim=-1).values, dim=-1).tolist()
# 输出结果
print("输入文本：", input_text)

print("下一个单词的前n个可能的候选结果和其对应的概率：")
for i in range(top_k):
    token = tokenizer.decode([top_k_tokens[i]], encoding="gb2312")
    probability = top_k_probabilities[i]
    print(f"{i + 1}. {token} - {probability:.4f}")
print("NEXT：", total_text.split(' ')[-1])

#
sample_idx = 2
input_text = val_datas[sample_idx]
total_text = val_datas[sample_idx + 1]
input_id_val = input_ids_val[sample_idx].unsqueeze(dim=0)
with torch.no_grad():
    outputs = model(input_id_val)
    predictions = outputs[0]
# 获取下一个单词的预测概率分布
next_token_predictions = predictions[0, -1, :]
print(next_token_predictions, len(next_token_predictions), max(next_token_predictions))
# 获取前k个最有可能的单词及其对应的概率
top_k_tokens = torch.topk(next_token_predictions, top_k, dim=-1).indices.tolist()
top_k_probabilities = torch.softmax(torch.topk(next_token_predictions, top_k, dim=-1).values, dim=-1).tolist()
# 输出结果
print("输入文本：", input_text)

print("下一个单词的前n个可能的候选结果和其对应的概率：")
ans_dict = {}
for i in range(top_k):
    token = tokenizer.decode([top_k_tokens[i]], encoding="gb2312")
    probability = top_k_probabilities[i]
    ans_dict[token] = (i + 1, probability)
    print(f"{i + 1}. {token} - {probability:.4f}")
ans = total_text.split(' ')[-1]
if ans in ans_dict:
    print("groud truth rank in ", ans_dict[ans][0], ans_dict[ans][1])
else:
    print("--mis in topk--")
print("groud truth：", ans)

# 定义token_vec的字典
token_embed = []
with open(args.pretrain_token_vecs_dict_path, "r", encoding='gb2312') as f:
    token_vecs_dict = json.load(f)
for word in vocab:
    token_embed.append(token_vecs_dict[word])
token_vecs = torch.tensor(token_embed)

# 定义CLS的字典
cls_labels = []
with open(args.pretrain_cls_labels_path, "r", encoding='gb2312') as f:
    cls_labels_dict = json.load(f)
for word in vocab:
    cls_labels.append(cls_labels_dict[word])
cls_labels = torch.tensor(cls_labels).unsqueeze(dim=1)

labels = ['6', '5', '4', '3', '2', '1', '0', '①', '②', '③', '④', '⑤', '⑥']
vol_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
def is_pred_valid_by_majority(pred_list, prob_list, threshold=1):
    label_prob_dict = {}
    for i in range(len(pred_list)):
        if pred_list[i] not in label_prob_dict:
            label_prob_dict[pred_list[i]] = prob_list[i]
        else:
            label_prob_dict[pred_list[i]] += prob_list[i]
    sorted_labels = sorted(label_prob_dict.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_labels)
    pred_close = sorted_labels[0][0][1]
    if abs(labels.index(pred_close) - labels.index('0')) <= threshold:
        return False, pred_close
    else:
        return True, pred_close

def is_pred_valid_by_mean(pred_list, prob_list, threshold=1):
    mean_idx = 0
    for i in range(len(pred_list)):
        pred_close = pred_list[i][1]
        mean_idx += prob_list[i] * labels.index(pred_close)
    pred_close = labels[int(mean_idx)]
    if abs(mean_idx - labels.index('0')) <= threshold:
        return False, pred_close
    else:
        return True, pred_close



def pred_old(sample_idx, data, data_ids):
    global top_k, word_weights_list
    input_text = data[sample_idx]
    total_text = data[sample_idx + 1]
    input = data_ids[sample_idx].unsqueeze(dim=0)
    with torch.no_grad():
        outputs = model(input)
        predictions = outputs[0]
    # 获取下一个单词的预测概率分布
    next_token_predictions = predictions[0, -1, :]  # * word_weights_list
    # print(next_token_predictions, len(next_token_predictions), max(next_token_predictions))
    # 获取前k个最有可能的单词及其对应的概率
    top_k_tokens = torch.topk(next_token_predictions, top_k, dim=-1).indices.tolist()
    top_k_probabilities = torch.softmax(torch.topk(next_token_predictions, top_k, dim=-1).values, dim=-1)

    top_k_probabilities_w = top_k_probabilities
    res_pair = dict(zip(list(top_k_tokens), list(top_k_probabilities_w)))
    res_pair = sorted(res_pair.items(), key=lambda x: x[1], reverse=True)
    res_pair = {k: v for k, v in res_pair}
    top_k_token_ids = list(res_pair.keys())
    top_k_tokens_old = [tokenizer.decode([top_k_token_ids[i]], encoding="gb2312") for i in range(top_k)]
    topk_vectors = [token_vecs_dict[t] for t in top_k_tokens_old]
    label_token = total_text.split(' ')[-1]
    label_vector = token_vecs_dict[label_token]
    sim_score = np.mean([np.linalg.norm(np.array(k_vector) - np.array(label_vector)) for k_vector in topk_vectors])
    # print([np.linalg.norm(np.array(k_vector) - np.array(label_vector)) for k_vector in topk_vectors])
    return sim_score, top_k_tokens_old, topk_vectors


def pred(sample_idx, data, data_ids):
    global top_k, word_weights_list
    input_text = data[sample_idx]
    total_text = data[sample_idx + 1]
    input = data_ids[sample_idx].unsqueeze(dim=0)
    with torch.no_grad():
        outputs = model(input)
        predictions = outputs[0]
    # 获取下一个单词的预测概率分布
    next_token_predictions = predictions[0, -1, :]  # * word_weights_list
    # print(next_token_predictions, len(next_token_predictions), max(next_token_predictions))
    # 获取前k个最有可能的单词及其对应的概率
    top_k_tokens = torch.topk(next_token_predictions, top_k, dim=-1).indices.tolist()
    top_k_probabilities = torch.softmax(torch.topk(next_token_predictions, top_k, dim=-1).values, dim=-1)

    word_weights_list_temp = torch.tensor([word_weights_list[t_idx] for t_idx in top_k_tokens])
    word_weights_list_temp = (word_weights_list_temp - torch.min(word_weights_list_temp)) / torch.max(
        word_weights_list_temp)
    top_k_probabilities_w = top_k_probabilities * word_weights_list_temp
    res_pair = dict(zip(list(top_k_tokens), list(top_k_probabilities_w)))
    res_pair = sorted(res_pair.items(), key=lambda x: x[1], reverse=True)
    res_pair = {k: v for k, v in res_pair}
    top_k_token_ids = list(res_pair.keys())
    top_k_probs = list(res_pair.values())
    top_k_tokens = [tokenizer.decode([top_k_token_ids[i]], encoding="gb2312") for i in range(top_k)]
    topk_vectors = [token_vecs_dict[t] for t in top_k_tokens]
    label_token = total_text.split(' ')[-1]
    label_vector = token_vecs_dict[label_token]
    sim_score = np.mean([np.linalg.norm(np.array(k_vector) - np.array(label_vector)) for k_vector in topk_vectors])
    # print([np.linalg.norm(np.array(k_vector) - np.array(label_vector)) for k_vector in topk_vectors])
    return sim_score, top_k_tokens, topk_vectors


def cls_check(sample_idx, data, data_ids):
    global top_k, word_weights_list
    input_text = data[sample_idx]
    total_text = data[sample_idx + 1]
    input = data_ids[sample_idx].unsqueeze(dim=0)
    with torch.no_grad():
        outputs = model(input)
        predictions = outputs[0]
    # 获取下一个单词的预测概率分布
    next_token_predictions = predictions[0, -1, :]  # * word_weights_list
    # print(next_token_predictions, len(next_token_predictions), max(next_token_predictions))
    # 获取前k个最有可能的单词及其对应的概率
    top_k_tokens = torch.topk(next_token_predictions, top_k, dim=-1).indices.tolist()
    top_k_probabilities = torch.softmax(torch.topk(next_token_predictions, top_k, dim=-1).values, dim=-1)

    top_k_probabilities_w = top_k_probabilities
    res_pair = dict(zip(list(top_k_tokens), list(top_k_probabilities_w)))
    res_pair = sorted(res_pair.items(), key=lambda x: x[1], reverse=True)
    res_pair = {k: v for k, v in res_pair}
    top_k_token_ids = list(res_pair.keys())
    probs = list(res_pair.values())
    top_k_tokens = [tokenizer.decode([top_k_token_ids[i]], encoding="gb2312") for i in range(top_k)]
    topk_labels = [cls_labels_dict[t] for t in top_k_tokens]
    label_token = total_text.split(' ')[-1]
    true_label = cls_labels_dict[label_token]
    score = [probs[i] if topk_labels[i] != -1 and topk_labels[i] == 1 else -probs[i] for i in range(len(topk_labels))]
    # score = [1 if label == true_label else 0 for label in topk_labels]
    rise_score = sum(score)
    # is_valid, pred_close = is_pred_valid_by_mean(top_k_tokens, probs, 1)
    # # is_valid, pred_close = is_pred_valid_by_majority(top_k_tokens, probs, 2)
    # if not is_valid:
    #     return -1
    if rise_score > 0 and true_label == 1 or rise_score < 0 and true_label == 0 or rise_score == 0 or label_token[1] == '0':
        ans = 1
    else :
        print(label_token)
        ans = 0
    return ans

import random

# sampled_idxs = random.sample(range(0, len(val_datas)), 100)
sampled_idxs = range(1, 1000)

total_mean_1 = []
total_cls_score_1 = []
for idx in sampled_idxs:
    top_k = 5
    mean_1, top_k_tokens_1, topk_vectors_1 = pred_old(idx, val_datas, input_ids_val)
    cls_score_1 = cls_check(idx, val_datas, input_ids_val)
    if cls_score_1 == -1:
        continue
    total_mean_1.append(mean_1)
    total_cls_score_1.append(cls_score_1)
    print("top-k sim score (v1)-step:", mean_1)
    print("top-k cls score (v1)-step:", cls_score_1)


print("length: ", len(total_cls_score_1))
print("top-k sim score (v1):", np.mean(total_mean_1))
print("top-k cls score (v1):", total_cls_score_1.count(1) / len(total_cls_score_1))
# print("top-k sim score (v2):", np.mean(total_mean_2))
