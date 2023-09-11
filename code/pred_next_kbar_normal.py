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
item_name = 'RM'
tick_name = '5min'
csv_file_path_test = '/data/users/lwb/pretrain_data/NORMAL/' + tick_name + 'Data_bin9/preprocessed_csvs/' + item_name + '/'
csv_file_path_val = '/data/users/lwb/pretrain_data/NORMAL/' + tick_name + 'Data_Val_bin9/preprocessed_csvs/' + item_name + '/'
label_dict = {
    "a": 1,
    "b": 1,
    "c": 2,
    "d": 2,
    "e": 2,
    "f": 2,
    "g": 2,
    "h": 2,
    "i": 2,
    "j": 3,
    "k": 3
}
ans_label_dict = {
    "a": 1,
    "b": 1,
    "c": 1,
    "d": 1,
    "e": 2,
    "f": 2,
    "g": 2,
    "h": 3,
    "i": 3,
    "j": 3,
    "k": 3
}
labels = ['6', '5', '4', '3', '2', '1', '0', '①', '②', '③', '④', '⑤', '⑥']
vol_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
hist_path = "/data/users/lwb/pretrain_data/NORMAL/bin_intervals/"
#
with open(hist_path + tick_name + "_vols_bins_dict.json", "r") as f:
    vols_dict = json.load(f)
with open(hist_path + tick_name + "_diff_opens_bins_dict.json", "r") as f:
    diff_opens_dict = json.load(f)
with open(hist_path + tick_name + "_diff_closes_bins_dict.json", "r") as f:
    diff_closes_dict = json.load(f)
with open(hist_path + tick_name + "_diff_highs_bins_dict.json", "r") as f:
    diff_highs_dict = json.load(f)
with open(hist_path + tick_name + "_diff_lows_bins_dict.json", "r") as f:
    diff_lows_dict = json.load(f)

item_vol_bins = vols_dict[item_name]
item_open_bins = diff_opens_dict[item_name]
item_close_bins = diff_closes_dict[item_name]
item_high_bins = diff_highs_dict[item_name]
item_low_bins = diff_lows_dict[item_name]


def get_label_interval_dict(labels, bins):
    label_interval_dict = {}
    for i in range(len(labels)):
        if bins[i] > 0 and bins[i + 1] > 0:
            bound = [bins[i], bins[i + 1]]
        elif bins[i] < 0 and bins[i + 1] < 0:
            bound = [bins[i], bins[i + 1]]
        else:
            bound = [bins[i], bins[i + 1]]
        # label_interval_dict[labels[i]] = [(bins[i], bins[i + 1]), bound]
        label_interval_dict[labels[i]] = bound
    return label_interval_dict


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


vol_label_interval_dict = get_label_interval_dict(list(label_dict.keys()), item_vol_bins)
open_label_interval_dict = get_label_interval_dict(list(label_dict.keys()), item_open_bins)
close_label_interval_dict = get_label_interval_dict(list(label_dict.keys()), item_close_bins)
high_label_interval_dict = get_label_interval_dict(list(label_dict.keys()), item_high_bins)
low_label_interval_dict = get_label_interval_dict(list(label_dict.keys()), item_low_bins)


def is_between(char, start, end):
    """
    判断字符是否在字母表中位于给定的两个字母之间
    :param char: 待判断的字符
    :param start: 起始字母
    :param end: 结束字母
    :return: True or False
    """
    if ord(start) <= ord(char) <= ord(end):
        return True
    else:
        return False


parser = ArgumentParser()
# parser.add_argument("--save_path", type=str, default="/data/users/lwb/ckpt/checkpoint-215000")
# parser.add_argument("--save_path", type=str, default="/data/users/lwb/cls_save/5min_gpt_model_cls_CU")
parser.add_argument("--save_path", type=str, default="/data/users/lwb/pretrain_data/NORMAL/5min-gpt-base-kbar-bin9-CLS")
# parser.add_argument("--save_path", type=str, default="/data/users/lwb/ckpt/5min-gpt-base-kbar-bin9-" + item_name)
args = parser.parse_args()

use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

# 加载tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("/data/users/lwb/pretrain_data/gpt2_tokenizer_bin9")
tokenizer = GPT2TokenizerKbar.from_pretrained("/data/users/lwb/pretrain_data/NORMAL/gpt2_tokenizer_bin9",
                                              encoding='gb2312')
tokenizer.pad_token = tokenizer.eos_token

# 加载预训练模型
model = GPT2LMHeadModel.from_pretrained(args.save_path)
model.config.pad_token_id = model.config.eos_token_id

# 生成文本
# 准备数据集：
st = 15
min_period_list = list(range(st, st + 1))

# test_save_path = '/data/users/lwb/pretrain_data/' + tick_name + 'Data_bin9_v44/datasets_for_items/' + item_name + '/'
# val_save_path = '/data/users/lwb/pretrain_data/' + tick_name + 'Data_bin9_Val_v44/datasets_for_items/' + item_name + '/' + "data_val.txt"
test_save_path = '/data/users/lwb/pretrain_data/NORMAL/' + tick_name + 'Data_bin9/datasets_for_items/' + item_name + '/'
total_save_path = '/data/users/lwb/pretrain_data/NORMAL/' + tick_name + 'Data_bin9/preprocessed_data/'
val_save_path = '/data/users/lwb/pretrain_data/NORMAL/' + tick_name + 'Data_Val_bin9/datasets_for_items/' + item_name + '/' + "data_val.txt"

# 读取数据并计算每个单词在原始数据集中出现的次数
filename = total_save_path + "5min.txt"
sentences = load_file(filename)
word_counts = count_words(sentences)
total_words = sum(word_counts.values())
# 使用TF-IDF分析器计算每个单词的权重
# sentences = [sentence.split(' ') for sentence in sentences]
# tfidf = TfidfVectorizer()
# tfidf.fit(sentences)
# print(len(tfidf.get_feature_names()))
# word_weights = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
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
frames_test = []
for file_name in os.listdir(csv_file_path_test):
    # 拼接文件路径
    file_path = os.path.join(csv_file_path_test, file_name)
    # 判断是否为文件
    if os.path.isfile(file_path):
        frame = pd.read_csv(file_path)
        if tick_name == "D":
            frame.sort_values('trade_date', inplace=True)
        else:
            frame.sort_values('trade_time', inplace=True)
        frame = frame.reset_index(drop=True)
        vol = frame['vol']
        frame.dropna(subset=['close'], inplace=True)
        frame.dropna(subset=['open'], inplace=True)
        frame.dropna(subset=['high'], inplace=True)
        frame.dropna(subset=['low'], inplace=True)
        frame.dropna(subset=['vol'], inplace=True)
        frame = frame[frame['vol'] >= 0]
        if frame.shape[0] <= 3:
            continue
        else:
            frames_test.append(frame)
print("test csv:", len(frames_test), len(test_data))

frames_val = []
for file_name in os.listdir(csv_file_path_val):
    # 拼接文件路径
    file_path = os.path.join(csv_file_path_val, file_name)
    # 判断是否为文件
    if os.path.isfile(file_path):
        frame = pd.read_csv(file_path)
        if tick_name == "D":
            frame.sort_values('trade_date', inplace=True)
        else:
            frame.sort_values('trade_time', inplace=True)
        frame = frame.reset_index(drop=True)
        vol = frame['vol']
        frame.dropna(subset=['close'], inplace=True)
        frame.dropna(subset=['open'], inplace=True)
        frame.dropna(subset=['high'], inplace=True)
        frame.dropna(subset=['low'], inplace=True)
        frame.dropna(subset=['vol'], inplace=True)
        frame = frame[frame['vol'] >= 0]
        if frame.shape[0] <= 3:
            continue
        else:
            frames_val.append(frame)
print("val csv:", len(frames_val), len(val_data[0:]))

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

# input_ids_val = input_ids_val[50000:]

sample_idx = 0
input_text = train_datas[sample_idx][5:]
total_text = train_datas[sample_idx + 1][5:]
input_id_train = input_ids_train[sample_idx].unsqueeze(dim=0)
# print(input_id_train)
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

valid_crr = 0
valid_total = 0
ent_threshold = 0.65
c_upper_thd = np.mean(close_label_interval_dict['i'])
c_lower_thd = np.mean(close_label_interval_dict['c'])
upper_thd = close_label_interval_dict['h'][0]
lower_thd = close_label_interval_dict['d'][1]

print(open_label_interval_dict)
print(close_label_interval_dict)
print(high_label_interval_dict)
print(low_label_interval_dict)


def text_to_interval(text, high_interval_dict, low_interval_dict):
    upper = high_interval_dict[text[2]][0]
    lower = low_interval_dict[text[3]][1]
    return [lower, upper]


def is_cls_correct(ans, preds, probs):
    global total_num, crr_num, valid_crr, valid_total
    total_num += 1
    score = 0
    pair = zip(preds, probs)
    pair = [str((close_label_interval_dict[p[0][1]][0] + close_label_interval_dict[p[0][1]][1]) / 2)[:7] for p in pair]
    ans_inter = text_to_interval(ans, high_label_interval_dict, low_label_interval_dict)
    for i in range(len(preds)):
        pred = preds[i]
        inter = close_label_interval_dict[pred[1]]
        if pred == '<|endoftext|>':
            continue
        prob = probs[i]
        if ans_inter[0] < (inter[0] + inter[1]) / 2 < ans_inter[1]:
            # if ans_inter[0] < inter[0] < ans_inter[1] and ans_inter[0] < inter[1] < ans_inter[1]:
            score += prob
        else:
            score -= prob
    if score > 0:
        crr_num += 1
    else:
        print(list(pair), [str(ans)[:7] for ans in ans_inter])
        valid_total += 1
    if total_num % 10 == 0:
        print("total acc rate: ", crr_num / total_num, " [" + str(total_num) + "]")


# scaler = MinMaxScaler()
def is_cls_correct_2(ans, preds, probs):
    global total_num, crr_num, valid_crr, valid_total
    total_num += 1
    score = 0
    labeled_scores = {'1': 0, '2': 0, '3': 0}
    probs = [round(prob, 5) for prob in probs]
    pair = list(zip(preds, probs))
    high_score_bottom = 0
    low_score_upper = 0
    high_score_upper = 0
    low_score_bottom = 0
    new_probs = []
    new_preds = []
    for i in range(len(preds)):
        pred = preds[i]
        if pred == '<|endoftext|>':
            continue
        else:
            new_preds.append(pred)
            new_probs.append(probs[i])
    for i in range(len(new_preds)):
        pred = new_preds[i]
        prob = new_probs[i] / sum(new_probs)
        high_score_bottom += prob * high_label_interval_dict[pred[2]][0]
        low_score_upper += prob * low_label_interval_dict[pred[3]][1]
        high_score_upper += prob * high_label_interval_dict[pred[2]][1]
        low_score_bottom += prob * low_label_interval_dict[pred[3]][0]
        # if label_dict[pred[1]] == 3 and label_dict[ans[2]] == 3 \
        #         or label_dict[pred[1]] == 1 and label_dict[ans[3]] == 1:
        #     score += prob
        # else:
        #     score -= prob
    # pred_label = max(labeled_scores, key=labeled_scores.get)
    # rise_fall_scores = [labeled_scores[label] for label in labeled_scores.keys() if label != '2']
    # entropy = stats.entropy(rise_fall_scores)
    if high_score_bottom > upper_thd and low_score_bottom > -upper_thd:
        pred_label = '1'
    elif low_score_upper < lower_thd and high_score_upper < -lower_thd:
        pred_label = '3'
    else:
        pred_label = '2'

    if pred_label == '1' or pred_label == '3':
        valid_total += 1
    if pred_label == '1' and high_label_interval_dict[ans[2]][0] >= upper_thd and low_label_interval_dict[
        ans[3]][0] > -upper_thd \
            or pred_label == '3' and low_label_interval_dict[ans[3]][1] <= lower_thd and high_label_interval_dict[
        ans[2]][1] < -lower_thd:
        valid_crr += 1
    else:
        pass
    if total_num % 100 == 0:
        # print("total acc rate: ", crr_num / total_num, " [" + str(total_num) + "]")
        if valid_total > 0:
            print(" valid acc rate: ", valid_crr / valid_total, " (" + str(valid_total) + "/", total_num, ")")
    # else:
    # print("---mistake---")


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
    print(next_token_predictions, len(next_token_predictions), max(next_token_predictions))
    # 获取前k个最有可能的单词及其对应的概率
    top_k_tokens = torch.topk(next_token_predictions, top_k, dim=-1).indices.tolist()
    top_k_probabilities = torch.softmax(torch.topk(next_token_predictions, top_k, dim=-1).values, dim=-1)

    top_k_probabilities_w = top_k_probabilities
    res_pair = dict(zip(list(top_k_tokens), list(top_k_probabilities_w)))
    res_pair = sorted(res_pair.items(), key=lambda x: x[1], reverse=True)
    res_pair = {k: v for k, v in res_pair}
    top_k_tokens = list(res_pair.keys())
    top_k_probabilities = list(res_pair.values())
    # 输出结果
    print("输入文本：", input_text)

    print("下一个单词的前n个可能的候选结果和其对应的概率：")
    ans_dict = {}
    preds = []
    probs = []
    candidates = []
    for i in range(top_k):
        token = tokenizer.decode([top_k_tokens[i]], encoding="gb2312")
        probability = top_k_probabilities[i]
        ans_dict[token] = (i + 1, probability)
        preds.append(token)
        probs.append(probability)
        candidates.append(token)
        print(f"{i + 1}. {token} - {probability:.4f}")
    ans = total_text.split(' ')[-1]
    if ans in ans_dict:
        print("groud truth rank in ", ans_dict[ans][0], ans_dict[ans][1])
    else:
        print("--mis prediction--")
    print("groud truth：", ans)
    return candidates


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
    print(next_token_predictions, len(next_token_predictions), max(next_token_predictions))
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
    top_k_tokens = list(res_pair.keys())
    top_k_probabilities = list(res_pair.values())
    # 输出结果
    print("输入文本：", input_text)

    print("下一个单词的前n个可能的候选结果和其对应的概率：")
    ans_dict = {}
    preds = []
    probs = []
    candidates = []
    for i in range(top_k):
        token = tokenizer.decode([top_k_tokens[i]], encoding="gb2312")
        probability = top_k_probabilities[i]
        ans_dict[token] = (i + 1, probability)
        preds.append(token)
        probs.append(probability)
        candidates.append(token)
        print(f"{i + 1}. {token} - {probability:.4f} - {word_weights[token]}")
    ans = total_text.split(' ')[-1]
    if ans in ans_dict:
        print("groud truth rank in ", ans_dict[ans][0], ans_dict[ans][1])
    else:
        print("--mis prediction--")
    print("groud truth：", ans)
    return candidates


def stat_preds(idx, data, data_ids):
    global top_k
    input_text = data[idx]
    total_text = data[idx + 1]
    if len(total_text) <= len(input_text):
        return
    input_ids = data_ids[idx].unsqueeze(dim=0)
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs[0]
    # 获取下一个单词的预测概率分布
    next_token_predictions = predictions[0, -1, :]
    # print(next_token_predictions, len(next_token_predictions), max(next_token_predictions))
    # 获取前k个最有可能的单词及其对应的概率
    top_k_tokens = torch.topk(next_token_predictions, top_k, dim=-1).indices.tolist()
    top_k_probabilities = torch.softmax(torch.topk(next_token_predictions, top_k, dim=-1).values, dim=-1).tolist()
    # 输出结果
    ans_dict = {}
    preds = []
    probs = []
    for i in range(top_k):
        token = tokenizer.decode([top_k_tokens[i]], encoding="gb2312")
        probability = top_k_probabilities[i]
        ans_dict[token] = (i + 1, probability)
        preds.append(token)
        probs.append(probability)
    ans = total_text.split(' ')[-1]
    is_cls_correct_2(ans, preds, probs)


total_num = 0
crr_num = 0

# for i in range(len(val_datas) - 1):  # train  _datas   val_datas input_ids_val
#     stat_preds(i, val_datas, input_ids_val)  # train_datas   val_datas input_ids_val
import utils

# item_vol_bins = vols_dict[item_name]
# item_open_bins = diff_opens_dict[item_name]
# item_close_bins = diff_closes_dict[item_name]
# item_high_bins = diff_highs_dict[item_name]
# item_low_bins = diff_lows_dict[item_name]


# labels = list(label_dict.keys())
label_vol_dict = {}
for i in range(len(item_vol_bins) - 1):
    if i == 0:
        mean_vol = item_vol_bins[i + 1]
    elif i == len(item_vol_bins) - 2:
        mean_vol = item_vol_bins[i]
    else:
        mean_vol = (item_vol_bins[i] + item_vol_bins[i + 1]) / 2
    label_vol_dict[vol_labels[i]] = mean_vol
print(label_vol_dict)

label_open_dict = {}
for i in range(len(item_open_bins) - 1):
    if i == 0:
        mean_vol = item_open_bins[i + 1]
    elif i == len(item_open_bins) - 2:
        mean_vol = item_open_bins[i]
    else:
        mean_vol = (item_open_bins[i] + item_open_bins[i + 1]) / 2
    label_open_dict[labels[i]] = mean_vol
print(label_open_dict)

label_close_dict = {}
for i in range(len(item_close_bins) - 1):
    if i == 0:
        mean_vol = item_close_bins[i + 1]
    elif i == len(item_close_bins) - 2:
        mean_vol = item_close_bins[i]
    else:
        mean_vol = (item_close_bins[i] + item_close_bins[i + 1]) / 2
    label_close_dict[labels[i]] = mean_vol
print(label_close_dict)

label_high_dict = {}
for i in range(len(item_high_bins) - 1):
    if i == 0:
        mean_vol = item_high_bins[i + 1]
    elif i == len(item_high_bins) - 2:
        mean_vol = item_high_bins[i]
    else:
        mean_vol = (item_high_bins[i] + item_high_bins[i + 1]) / 2
    label_high_dict[labels[i]] = mean_vol
print(label_high_dict)

label_low_dict = {}
for i in range(len(item_low_bins) - 1):
    if i == 0:
        mean_vol = item_low_bins[i + 1]
    elif i == len(item_low_bins) - 2:
        mean_vol = item_low_bins[i]
    else:
        mean_vol = (item_low_bins[i] + item_low_bins[i + 1]) / 2
    label_low_dict[labels[i]] = mean_vol
print(label_low_dict)

while (1):
    top_k = 100
    pic_top_k = 10
    idx = int(input("idx："))
    if idx == "":
        break
    else:
        if val_csv_dict[idx] != val_csv_dict[idx + 1]:
            print("re input...")
            continue
        cands_1 = pred(idx, val_datas, input_ids_val)
        cands_2 = pred_old(idx, val_datas, input_ids_val)

        # vis
        texts = val_datas[idx + 1].split(' ')
        # frame = frames_val[val_csv_dict[idx]]
        # pre_close_list = frame["pre_close"].tolist()[:len(texts)]
        pre_close_list = [1000]
        for i in range(0, len(texts) - 1):
            pre_close_list.append(pre_close_list[-1] * (1 + label_close_dict[texts[i][1]]))
        pd_true = utils.text_kbar(
            texts,
            pre_close_list,
            label_vol_dict,
            label_open_dict,
            label_close_dict,
            label_high_dict,
            label_low_dict,
        )
        utils.plot_kline(pd_true, '/data/users/lwb/pretrain_data/NORMAL/pics/true')
        # mpf.plot(data, type='candle', mav=(), volume=True, savefig='/data/users/lwb/pretrain_data/UNIFORMED/pics/true.png')

        # 1
        for text in cands_1[:pic_top_k]:
            t_texts = texts[: -1] + [text]
            print("frames_val:", len(frames_val))
            # frame = frames_val[val_csv_dict[idx]]
            # pre_close_list = frame["pre_close"].tolist()[:len(t_texts)]
            pre_close_list = [1000]
            for i in range(0, len(texts) - 1):
                pre_close_list.append(pre_close_list[-1] * (1 + label_close_dict[t_texts[i][1]]))
            pd_1 = utils.text_kbar(
                t_texts,
                pre_close_list,
                label_vol_dict,
                label_open_dict,
                label_close_dict,
                label_high_dict,
                label_low_dict,
            )
            utils.plot_kline(pd_1, '/data/users/lwb/pretrain_data/NORMAL/pics/cand1_' + text)
        # pd_1.to_csv('/data/users/lwb/pretrain_data/NORMAL/pics/cand1_' + text + '.csv', index=False)
        # utils.plot_kline(frame.head(len(t_texts)), '/data/users/lwb/pretrain_data/UNIFORMED/pics/cand1_' + text + '')
        # mpf.plot(data, type='candle', mav=(), volume=True,
        #          savefig='/data/users/lwb/pretrain_data/UNIFORMED/pics/cand1_' + text + '.png')
        # 2

        for text in cands_2[:pic_top_k]:
            t_texts = texts[: -1] + [text]
            # frame = frames_val[val_csv_dict[idx]]
            # pre_close_list = frame["pre_close"].tolist()[:len(t_texts)]
            pre_close_list = [1000]
            for i in range(0, len(texts) - 1):
                pre_close_list.append(pre_close_list[-1] * (1 + label_close_dict[t_texts[i][1]]))
            pd_2 = utils.text_kbar(
                t_texts,
                pre_close_list,
                label_vol_dict,
                label_open_dict,
                label_close_dict,
                label_high_dict,
                label_low_dict,
            )
            utils.plot_kline(pd_2, '/data/users/lwb/pretrain_data/NORMAL/pics/cand2_' + text)
            # mpf.plot(data, type='candle', mav=(), volume=True,
            #          savefig='/data/users/lwb/pretrain_data/UNIFORMED/pics/cand2_' + text + '.png')
