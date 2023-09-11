import json
import os
import string
from argparse import ArgumentParser
from collections import Counter

import constants
import matplotlib.pyplot as plt
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

item_dict = constants.item_dict
pos_count = 0
total_count = 0
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
labels = ['6', '5', '4', '3', '2', '1', '0', '①', '②', '③', '④', '⑤', '⑥']
vol_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

parser = ArgumentParser()
# parser.add_argument("--save_path", type=str,
#                     default="/INPUT/lwb/k-emebedding/NORMAL/5min-gpt-base-kbar-bin9-SIM(epoch-200)")

parser.add_argument("--save_path", type=str,
                    default="/INPUT/lwb/k-emebedding/NORMAL/checkpoint_kbar_bin13_SIM_H5_total-lseq/checkpoint-430000")

# parser.add_argument("--save_path", type=str,
#                     default="/INPUT/lwb/k-emebedding/NORMAL/5min-gpt-base-kbar-bin9-CLS(epoch-300)")
# parser.add_argument("--save_path", type=str, default="/INPUT/lwb/k-emebedding/NORMAL/5min-gpt-base-kbar-bin9-CLS(epoch-200)")
# parser.add_argument("--save_path", type=str, default="/INPUT/lwb/k-emebedding/NORMAL/5min-gpt-base-kbar-bin9(epoch-200)")
# parser.add_argument("--save_path", type=str, default="/INPUT/lwb/k-emebedding/NORMAL/5min-gpt-base-kbar-bin9-CLS-SIM(epoch-200)")

# Tokenizer
# parser.add_argument("--pretrain_cls_labels_path", type=str,
#                     default="/INPUT/lwb/k-emebedding/NORMAL/5minData_bin9/preprocessed_data/cls_labels.json")
parser.add_argument("--pretrain_cls_labels_path", type=str,
                    default="/INPUT/lwb/k-emebedding/NORMAL/gpt2_tokenizer_bin13_H5/cls_labels.json")
parser.add_argument("--tokenizer_path", type=str,
                    default="/INPUT/lwb/k-emebedding/NORMAL/gpt2_tokenizer_bin13_H5")
args = parser.parse_args()


tokenizer = GPT2TokenizerKbar.from_pretrained(args.tokenizer_path, encoding='gb2312')
tokenizer.pad_token = tokenizer.eos_token
vocab = tokenizer.get_vocab()
# 加载预训练模型
model = GPT2LMHeadModel.from_pretrained(args.save_path)
model.config.pad_token_id = model.config.eos_token_id

# 定义CLS的字典
cls_labels = []
with open(args.pretrain_cls_labels_path, "r", encoding='gb2312') as f:
    cls_labels_dict = json.load(f)
for word in vocab:
    cls_labels.append(cls_labels_dict[word])
cls_labels = torch.tensor(cls_labels).unsqueeze(dim=1)
tokens_list = []
prices_list = []
trade_time_list = []

# bin 数据加载
tick_name = '5min'
hist_path = "/INPUT/lwb/k-emebedding/NORMAL/bin_intervals/"
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


def prepare_data(item_name):
    global tokens_list, prices_list, trade_time_list
    tokens_list = []
    prices_list = []
    trade_time_list = []
    print("prepare data..")
    csv_file_path_val = '/INPUT/lwb/k-emebedding/NORMAL/' + tick_name + 'Data_Val_bin9/preprocessed_csvs/' + item_name + '/'

    vol_chg_bins = vols_dict[item_name]
    diff_opens_bins = diff_opens_dict[item_name]
    diff_closes_bins = diff_closes_dict[item_name]
    diff_highs_bins = diff_highs_dict[item_name]
    diff_lows_bins = diff_lows_dict[item_name]

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

            if frame.shape[0] == 0:
                continue
            frame['vol_dcr'] = pd.cut(frame['vol'], bins=vol_chg_bins, labels=vol_labels)
            frame['open_dcr'] = pd.cut(frame['diff_open'], bins=diff_opens_bins, labels=labels)
            frame['close_dcr'] = pd.cut(frame['diff_close'], bins=diff_closes_bins, labels=labels)
            frame['high_dcr'] = pd.cut(frame['diff_high'], bins=diff_highs_bins, labels=labels)
            frame['low_dcr'] = pd.cut(frame['diff_low'], bins=diff_lows_bins, labels=labels)
            frame['kbar_id'] = frame['open_dcr'].astype('str') + frame['close_dcr'].astype('str') + frame[
                'high_dcr'].astype('str') \
                               + frame['low_dcr'].astype('str') + frame['vol_dcr'].astype('str')
            bars = list(frame['kbar_id'])
            opens = list(frame["open"])
            closes = list(frame["close"])
            highs = list(frame["high"])
            lows = list(frame["low"])
            prices = [[opens[i], closes[i], highs[i], lows[i]] for i in range(len(opens))]
            if tick_name == "D":
                times = list(frame['trade_date'])
            else:
                times = list(frame['trade_time'])[0]
            tokens_list.append(bars)
            prices_list.append(prices)
            trade_time_list.append(times)
    zipped = list(zip(tokens_list, prices_list, trade_time_list))
    sorted_zipped = sorted(zipped, key=lambda x: x[2])
    tokens_list, prices_list, trade_time_list = zip(*sorted_zipped)


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


def cls_check(input_tokens, top_k=5, label_token=None):
    input = torch.tensor(tokenizer.convert_tokens_to_ids(input_tokens)).unsqueeze(dim=0)
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
    true_label = cls_labels_dict[label_token]
    score = [probs[i] if topk_labels[i] != -1 and topk_labels[i] == 1 else -probs[i] for i in range(len(topk_labels))]
    # score = [1 if label == true_label else 0 for label in topk_labels]
    rise_score = sum(score)
    # is_valid, pred_close = is_pred_valid_by_mean(top_k_tokens, probs, 2)
    # # is_valid, pred_close = is_pred_valid_by_majority(top_k_tokens, probs, 2)
    # if not is_valid:
    #     return -1
    if rise_score > 0 and true_label == 1 or rise_score < 0 and true_label == 0 or rise_score == 0 or label_token[
        1] == '0':
        ans = 1
    else:
        print(label_token)
        ans = 0
    return ans


def is_pred_valid_by_majority(pred_list, prob_list, threshold=1, last_close=None, check_next=None):
    label_prob_dict = {}
    rise_fall_freq_dict = {"fall": 0, "steady": 0, "rise": 0}
    rise_fall_sum = 0
    for i in range(len(pred_list)):
        rise_fall_sum += prob_list[i]
        if labels.index(pred_list[i][1]) < labels.index("0"):
            rise_fall_freq_dict['fall'] += prob_list[i]
        elif labels.index(pred_list[i][1]) > labels.index("0"):
            rise_fall_freq_dict["rise"] += prob_list[i]
        else:
            rise_fall_freq_dict["steady"] += prob_list[i]
        if pred_list[i] not in label_prob_dict:
            label_prob_dict[pred_list[i]] = round(prob_list[i], 4)
        else:
            label_prob_dict[pred_list[i]] += round(prob_list[i], 4)
    sorted_labels = sorted(label_prob_dict.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_labels, check_next)
    pred_close = sorted_labels[0][0][1]
    # print(sorted_labels)
    # pred_close = sorted_labels[0][0][1]
    # print(sorted_labels)
    # print(sorted_labels[0][0][1])
    threshold_check = abs(labels.index(pred_close) - labels.index('0')) > threshold
    # print(threshold_check)
    entropy_check = rise_fall_freq_dict["rise"] > 0.7 * rise_fall_sum or rise_fall_freq_dict[
        "fall"] > 0.7 * rise_fall_sum
    # print(entropy_check, rise_fall_freq_dict)
    if last_close is not None:
        vol_check = vol_labels.index(last_close) == vol_labels.index('e')
        # print(vol_check, last_close)
    else:
        vol_check = True
    if threshold_check and vol_check:
        return True, pred_close
    else:
        return False, pred_close


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


def get_a_signal_test(input_tokens, top_k=5, threshold=1, check_next=None, diff_price=None):
    input = torch.tensor(tokenizer.convert_tokens_to_ids(input_tokens)).unsqueeze(dim=0)
    with torch.no_grad():
        outputs = model(input)
        predictions = outputs[0]
    # 获取下一个单词的预测概率分布
    next_token_predictions = predictions[0, -1, :]  # * word_weights_list
    # print(next_token_predictions, len(next_token_predictions), max(next_token_predictions))
    # 获取前k个最有可能的单词及其对应的概率
    top_k_token_ids = torch.topk(next_token_predictions, top_k, dim=-1).indices.tolist()
    probs = torch.softmax(torch.topk(next_token_predictions, top_k, dim=-1).values, dim=-1).tolist()
    top_k_tokens = [tokenizer.decode([top_k_token_ids[i]], encoding="gb2312") for i in range(top_k)]
    # is_valid, pred_close = is_pred_valid_by_majority(top_k_tokens, probs, threshold, input_tokens[-1][4], check_next=check_next)
    topk_labels = [cls_labels_dict[t] for t in top_k_tokens]
    score = [probs[i] if topk_labels[i] != -1 and topk_labels[i] == 1 else -probs[i] for i in range(len(topk_labels))]
    rise_score = sum(score)
    signal = ""
    if rise_score < 0:
        signal = "sell"
    elif rise_score > 0:
        signal = "buy"
    else:
        pass
    true_label = cls_labels_dict[check_next]
    if rise_score > 0 and diff_price >= 0 or rise_score < 0 and diff_price <= 0:
        ans = 1
    else:
        if check_next[1] == '0':
            ans = 1
        else:
            print(check_next)
            ans = 0
    # if is_valid:
    #     if labels.index(pred_close) < labels.index('0'):
    #         signal = "sell"
    #     else:
    #         signal = "buy"
    # else:
    #     pass
    return signal, ans


def get_a_signal_3(input_tokens, pre_signal, top_k=5, threshold=1.0, check_next=None):
    if pre_signal == "buy":
        pre_label = 1
    elif pre_signal == "sell":
        pre_label = 0
    else:
        pre_label = -10
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(input_tokens)).unsqueeze(dim=0)
    # 使用generate方法生成接下来的两个单词，设置num_return_sequences为5表示返回5个候选序列
    # print(input_ids)
    # print(top_k)
    output_sequences = model.generate(input_ids,
                                      max_length=input_ids.shape[-1] + 3,
                                      num_return_sequences=top_k,
                                      num_beams=top_k,
                                      num_beam_groups=1)
    ## 预测两个bar
    # print(output_sequences)
    output_sequence = output_sequences[0 * top_k: (0 + 1) * top_k]
    # 获取最后两个单词的token id
    last_three_token_ids = output_sequence[:, -3:].tolist()
    # 使用torch.topk()方法一次性获取每个位置上前5个最有可能的单词及其对应的概率
    top_k_token_ids, top_k_probs = torch.topk(model(input_ids[0].unsqueeze(0))[0][0, -1, :], 5, dim=-1)
    # top_k_token_ids = top_k_token_ids.tolist()
    last_three_tokens_list = []
    for token_id_pair in last_three_token_ids:
        last_three_tokens_list.append([tokenizer.decode(token_id_pair[0]),
                                       tokenizer.decode(token_id_pair[1]),
                                       tokenizer.decode(token_id_pair[2])])
    probs = top_k_probs.tolist()

    # topk个接下来的三个词的labels，只保留label相同的，不同的话则记为-10
    topk_seq_labels = [
        cls_labels_dict[t[0]] if cls_labels_dict[t[0]] == cls_labels_dict[t[1]] and cls_labels_dict[t[0]] ==
                                 cls_labels_dict[t[1]] else -1 for t in last_three_tokens_list]
    # # topk个接下来的一个词的labels, 只保留与pre_signal一致的，不同的话记为-10
    # topk_seq1st_labels = [cls_labels_dict[t[0]] if cls_labels_dict[t[0]] == pre_label else -1 for t in last_two_tokens_list]
    topk_seq1st_labels = [cls_labels_dict[t[0]] for t in last_three_tokens_list]

    if pre_signal == "":
        if topk_seq_labels.count(-1) == len(topk_seq_labels):
            signal = ""
        else:
            score = [probs[i] if topk_seq_labels[i] != -1 and topk_seq_labels[i] == 1 else -probs[i] for i in
                     range(len(topk_seq_labels))]
            rise_score = sum(score)
            if rise_score < -threshold:
                signal = "sell"
            elif rise_score > threshold:
                signal = "buy"
            else:
                signal = ""
    else:
        score = [probs[i] if topk_seq1st_labels[i] != -1 and topk_seq1st_labels[i] == 1 else -probs[i] for i in
                 range(len(topk_seq1st_labels))]
        rise_score = sum(score)
        if rise_score < -threshold:
            signal = "sell"
        elif rise_score > threshold:
            signal = "buy"
        else:
            signal = ""
        if signal != pre_signal:
            if topk_seq_labels.count(-1) == len(topk_seq_labels):
                signal = ""
            else:
                score = [probs[i] if topk_seq_labels[i] != -1 and topk_seq_labels[i] == 1 else -probs[i] for i in
                         range(len(topk_seq_labels))]
                rise_score = sum(score)

                if rise_score < -threshold:
                    # print(score, sum(score))
                    signal = "sell"
                elif rise_score > threshold:
                    # print(score, sum(score))
                    signal = "buy"
                else:
                    signal = ""
        else:
            pass
    return signal


def get_a_signal_2(input_tokens, pre_signal, top_k=5, threshold=1.0, check_next=None):
    if pre_signal == "buy":
        pre_label = 1
    elif pre_signal == "sell":
        pre_label = 0
    else:
        pre_label = -10
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(input_tokens)).unsqueeze(dim=0)
    # 使用generate方法生成接下来的两个单词，设置num_return_sequences为5表示返回5个候选序列
    # print(input_ids)
    # print(top_k)
    output_sequences = model.generate(input_ids,
                                      max_length=input_ids.shape[-1] + 2,
                                      num_return_sequences=top_k,
                                      num_beams=top_k,
                                      num_beam_groups=1)
    ## 预测两个bar
    # print(output_sequences)
    output_sequence = output_sequences[0 * top_k: (0 + 1) * top_k]
    # 获取最后两个单词的token id
    last_two_token_ids = output_sequence[:, -2:].tolist()
    # 使用torch.topk()方法一次性获取每个位置上前5个最有可能的单词及其对应的概率
    top_k_token_ids, top_k_probs = torch.topk(model(input_ids[0].unsqueeze(0))[0][0, -1, :], 5, dim=-1)
    # top_k_token_ids = top_k_token_ids.tolist()
    last_two_tokens_list = [[tokenizer.decode(token_id_pair[0]), tokenizer.decode(token_id_pair[1])] for token_id_pair
                            in last_two_token_ids]
    probs = top_k_probs.tolist()

    # topk个接下来的两个词的labels，只保留label相同的，不同的话则记为-10
    topk_seq_labels = [cls_labels_dict[t[0]] if cls_labels_dict[t[0]] == cls_labels_dict[t[1]] else -1 for t in
                       last_two_tokens_list]
    # # topk个接下来的一个词的labels, 只保留与pre_signal一致的，不同的话记为-10
    # topk_seq1st_labels = [cls_labels_dict[t[0]] if cls_labels_dict[t[0]] == pre_label else -1 for t in last_two_tokens_list]
    topk_seq1st_labels = [cls_labels_dict[t[0]] for t in last_two_tokens_list]

    if pre_signal == "":
        if topk_seq_labels.count(-1) == len(topk_seq_labels):
            signal = ""
        else:
            score = [probs[i] if topk_seq_labels[i] != -1 and topk_seq_labels[i] == 1 else -probs[i] for i in
                     range(len(topk_seq_labels))]
            rise_score = sum(score)
            if rise_score < -threshold:
                signal = "sell"
            elif rise_score > threshold:
                signal = "buy"
            else:
                signal = ""
    else:
        score = [probs[i] if topk_seq1st_labels[i] != -1 and topk_seq1st_labels[i] == 1 else -probs[i] for i in
                 range(len(topk_seq1st_labels))]
        rise_score = sum(score)
        if rise_score < -threshold:
            signal = "sell"
        elif rise_score > threshold:
            signal = "buy"
        else:
            signal = ""
        if signal != pre_signal:
            if topk_seq_labels.count(-1) == len(topk_seq_labels):
                signal = ""
            else:
                score = [probs[i] if topk_seq_labels[i] != -1 and topk_seq_labels[i] == 1 else -probs[i] for i in
                         range(len(topk_seq_labels))]
                rise_score = sum(score)

                if rise_score < -threshold:
                    # print(score, sum(score))
                    signal = "sell"
                elif rise_score > threshold:
                    # print(score, sum(score))
                    signal = "buy"
                else:
                    signal = ""
        else:
            pass
    return signal


def get_a_signal(input_tokens, previous_tokens, pre_signal=None, top_k=5, threshold=1.0, check_next=None):
    # print(previous_tokens)
    # print(input_tokens)
    input = torch.tensor(tokenizer.convert_tokens_to_ids(previous_tokens + input_tokens)).unsqueeze(dim=0)

    # 预测一个bar
    # print("predicting signals...")
    with torch.no_grad():
        outputs = model(input)
        predictions = outputs[0]
    # 获取下一个单词的预测概率分布
    next_token_predictions = predictions[0, -1, :]  # * word_weights_list
    # 获取前k个最有可能的单词及其对应的概率
    top_k_token_ids = torch.topk(next_token_predictions, 10, dim=-1).indices.tolist()[:5]
    probs = torch.softmax(torch.topk(next_token_predictions, 10, dim=-1).values, dim=-1).tolist()[:5]
    top_k_tokens = [tokenizer.decode([top_k_token_ids[i]], encoding="gb2312") for i in range(top_k)]

    # is_valid, pred_close = is_pred_valid_by_majority(top_k_tokens, probs, threshold, input_tokens[-1][4], check_next=check_next)
    topk_labels = [cls_labels_dict[t] for t in top_k_tokens]
    score = [probs[i] if topk_labels[i] != -1 and topk_labels[i] == 1 else -probs[i] for i in range(len(topk_labels))]

    rise_score = sum(score)
    signal = ""
    if rise_score < -threshold:
        signal = "sell"
    elif rise_score > threshold:
        signal = "buy"
    else:
        pass
    return signal


def get_signals(data_ids, previous_tokens, top_k=5, threshold=1.0, st=15):
    global total_ans, pos_ans
    signals = [""] * len(data_ids)
    for i in range(st, len(data_ids) - 1):
        input_tokens = data_ids[:i]
        signal = get_a_signal(input_tokens, previous_tokens, pre_signal=signals[i - 2], top_k=top_k, threshold=threshold,
                              check_next=data_ids[i])
        signals[i - 1] = signal
    return signals


def compute_an_earning(prices, signals, fees_ratio, t_name):
    global total_count, pos_count, t_fee_dict
    earning = 0
    if t_name in t_fee_dict.keys():
        daily_trans_fee = len(merge_adjacent(signals)) * t_fee_dict[t_name]
    else:
        daily_trans_fee = len(merge_adjacent(signals)) * prices[0][0] * fees_ratio * 2
    for i in range(len(signals)):
        total_count += 1

        # print(total_count, pos_count)
        if signals[i] == "buy":
            earning += prices[i + 1][1] - prices[i][1]
            if prices[i + 1][1] - prices[i][1] >= 0:
                pos_count += 1
            # print(signals[i], ": ", prices[i + 1][1] - prices[i][1])
        elif signals[i] == "sell":
            earning += prices[i][1] - prices[i + 1][1]
            if prices[i][1] - prices[i + 1][1] >= 0:
                pos_count += 1
            # print(signals[i], ": ", prices[i][1] - prices[i + 1][1])
        else:
            total_count -= 1
    print("Number of daily transactions :", 2 * len(merge_adjacent(signals)))
    # print("Daily fees :", round(daily_fees,2))
    return earning - daily_trans_fee, daily_trans_fee


def merge_adjacent(input_list):
    m_list = [item for item in input_list if item != '']
    # print(m_list)
    for i in range(len(m_list) - 1, 0, -1):
        if m_list[i] == m_list[i - 1]:
            del m_list[i]
    # print(m_list)
    return m_list


def compute_earnings(data_ids_list, t_name, previous_tokens, fees_ratio=0.0, top_k=5, threshold=1.0, st=15, pre_thr=100):
    earnings = []
    # print("compute earnings...")
    for i in range(len(data_ids_list)):
        signals = get_signals(data_ids_list[i], previous_tokens, top_k, threshold, st)
        earning, fees = compute_an_earning(prices_list[i], signals, fees_ratio, t_name)
        if earning < 0:
            print("return: ", round(earning, 3), "trans fees:", round(fees, 3), " time:", trade_time_list[i])
        else:
            print("return: ", round(earning, 3), "trans fees:", round(fees, 3))
        earnings.append(earning)
        previous_tokens += data_ids_list[i]
        previous_tokens = previous_tokens[-pre_thr:]
    return earnings


if __name__ == "__main__":
    path = '/INPUT/lwb/k-emebedding/NORMAL/5minData_Val_bin9/preprocessed_csvs'
    t_fee_dict = {"RU": 0.3, "M": 0.3}
    # 获取目录下的所有子目录名
    item_names = [subdir for subdir in os.listdir(path) if os.path.isdir(os.path.join(path, subdir))]
    saved_names = list(vols_dict.keys())
    item_names = list(set(item_names).intersection(set(saved_names)))
    print(item_names)
    for item_name in item_names:
        print("=====================   ", item_name, "   =====================")
        previous_tokens = []
        prepare_data(item_name)

        png_save_path = "/home/lwb/Kbar/pngs/"
        position_ratio = 3
        base = prices_list[0][0][0] * position_ratio
        levearage = 10
        fees_ratio = 0.00000 + 0.000000
        print(base)
        # total_cls_score_1 = []
        # for i in range(len(tokens_list)):
        #     for j in range(15, len(tokens_list[i]) - 1):
        #         top_k = 5
        #         cls_score_1 = cls_check(tokens_list[i][:j], 5, tokens_list[i][j + 1])
        #         if cls_score_1 == -1:
        #             continue
        #         total_cls_score_1.append(cls_score_1)
        #         print("top-k cls score (v1)-step:", cls_score_1)
        # print("length: ", len(total_cls_score_1))
        # print("top-k cls score (v1):", total_cls_score_1.count(1) / len(total_cls_score_1))
        # print("--------------------------------")

        # total_cls_score_1 = []
        # for i in range(len(tokens_list)):
        #     for j in range(15, len(tokens_list[i]) - 1):
        #         top_k = 5
        #         diff_price = prices_list[i][j+1][1] - prices_list[i][j][1]
        #         _, cls_score_1 = get_a_signal_test(tokens_list[i][:j], 5, 1, tokens_list[i][j + 1], diff_price)
        #         if cls_score_1 == -1:
        #             continue
        #         total_cls_score_1.append(cls_score_1)
        #         print("top-k cls score (v1)-step:", cls_score_1)
        # print("length: ", len(total_cls_score_1))
        # print("top-k cls score (v1):", total_cls_score_1.count(1) / len(total_cls_score_1))
        # print("--------------------------------")

        print("length:", len(tokens_list))
        return_list = compute_earnings(tokens_list,
                                       item_name,
                                       previous_tokens=previous_tokens,
                                       fees_ratio=fees_ratio,
                                       top_k=5,
                                       threshold=0.0,
                                       st=15,
                                       pre_thr=15)
        return_list = [r * levearage for r in return_list]
        count = len(list(filter(lambda x: x >= 0, return_list)))
        print("good decision num: ", count, "/", len(return_list), "-", pos_count / total_count)
        print("total return: ", sum(return_list))

        # 计算当日累计收益率
        cumulative_returns = [1]
        for r in return_list:
            cumulative_returns.append(cumulative_returns[-1] * (1 + r / base))


        # 计算年化收益率
        annualized_returns = (cumulative_returns[-1] ** (250 / len(return_list))) - 1
        print('Annualized Returns: {:.2%}'.format(annualized_returns))

        # 计算最大回撤率
        max_drawdown = 0
        peak = cumulative_returns[0]
        for r in cumulative_returns:
            if r > peak:
                peak = r
            drawdown = (peak - r) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        print('Max Drawdown: {:.2%}'.format(max_drawdown))

        # 绘制收益曲线图
        plt.plot(cumulative_returns)
        plt.title('Cumulative Returns')
        plt.xlabel('Day')
        plt.ylabel('Cumulative Returns')
        plt.text(0.05, 0.95,
                 'Annualized Returns: {:.2%}'.format(annualized_returns) + '\n' + 'Max Drawdown: {:.2%}'.format(
                     max_drawdown) \
                 + '\nLeverage ratio: ' + str(levearage) + 'x\nPosition ratio: ' + str(
                     round(1 / position_ratio * 100, 2)) + '%',
                 fontsize=12,
                 color='red',
                 bbox={'facecolor': 'yellow', 'pad': 10},
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform=plt.gca().transAxes)
        plt.savefig(png_save_path + item_name + "Cumulative Returns.png", format='png', dpi=300)
        plt.close()
