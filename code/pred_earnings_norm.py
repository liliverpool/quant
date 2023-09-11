import json
import os
import string
from argparse import ArgumentParser
import datetime
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

# price bin
labels = constants.labels_13
price_bins = constants.price_diff_bins_13

# volume bin
vol_labels = constants.vol_labels_11
tick_name = '5min'
hist_path = "/INPUT/lwb/k-emebedding/NORMAL/bin_intervals/"
with open(hist_path + tick_name + "_vols_bins_dict.json", "r") as f:
    vols_dict = json.load(f)


parser = ArgumentParser()
# parser.add_argument("--save_path", type=str,
#                     default="/INPUT/lwb/k-emebedding/NORMAL/5min-gpt-base-kbar-bin9-SIM(epoch-200)")
parser.add_argument("--norm_max_path_H5", type=str, default='/INPUT/lwb/k-emebedding/NORMAL/5minData_Val_bin9/norm_max/item_max_H5.json')
parser.add_argument("--norm_max_path_H15", type=str, default='/INPUT/lwb/k-emebedding/NORMAL/5minData_Val_bin9/norm_max/item_max_H15.json')
parser.add_argument("--norm_max_path_H30", type=str, default='/INPUT/lwb/k-emebedding/NORMAL/5minData_Val_bin9/norm_max/item_max_H30.json')
parser.add_argument("--norm_max_path_H60", type=str, default='/INPUT/lwb/k-emebedding/NORMAL/5minData_Val_bin9/norm_max/item_max_H60.json')
parser.add_argument("--norm_max_path_daily", type=str, default='/INPUT/lwb/k-emebedding/NORMAL/5minData_Val_bin9/norm_max/item_max_daily.json')
parser.add_argument("--save_path", type=str,
                    default="/INPUT/lwb/k-emebedding/NORMAL/H-5-gpt-base-kbar-bin13-SIM")
# parser.add_argument("--save_path", type=str, default="/INPUT/lwb/k-emebedding/NORMAL/5min-gpt-base-kbar-bin9-CLS(epoch-200)")
# parser.add_argument("--save_path", type=str, default="/INPUT/lwb/k-emebedding/NORMAL/5min-gpt-base-kbar-bin9(epoch-200)")
# parser.add_argument("--save_path", type=str, default="/INPUT/lwb/k-emebedding/NORMAL/5min-gpt-base-kbar-bin9-CLS-SIM(epoch-200)")
parser.add_argument("--pretrain_cls_labels_path", type=str,
                    default="/INPUT/lwb/k-emebedding/NORMAL/gpt2_tokenizer_bin13_H5/cls_labels.json")
args = parser.parse_args()
# 加载tokenizer
tokenizer = GPT2TokenizerKbar.from_pretrained("/INPUT/lwb/k-emebedding/NORMAL/gpt2_tokenizer_bin13_H5", encoding='gb2312')
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

# 加载norm max
with open(args.norm_max_path_H5, "r", encoding='gb2312') as f:
    norm_max_H5 = json.load(f)
with open(args.norm_max_path_H15, "r", encoding='gb2312') as f:
    norm_max_H15 = json.load(f)
with open(args.norm_max_path_H30, "r", encoding='gb2312') as f:
    norm_max_H30 = json.load(f)
with open(args.norm_max_path_H60, "r", encoding='gb2312') as f:
    norm_max_H60 = json.load(f)
with open(args.norm_max_path_daily, "r", encoding='gb2312') as f:
    norm_max_daily = json.load(f)


def get_last_min_time(trade_time, delta_min):
    min_time = trade_time.minute
    last_time = trade_time - datetime.timedelta(minutes=delta_min + min_time)
    return last_time


def get_last_daily_time(trade_time):
    last_time = trade_time - datetime.timedelta(days=1)
    while last_time.weekday() == 5 or last_time.weekday() == 6:
        last_time = last_time - datetime.timedelta(days=1)
    last_time = last_time.replace(hour=8)
    last_time = last_time.replace(minute=55)
    last_time = last_time.replace(second=0)
    return last_time


def to_daily_start(date_time, delta_days=0):
    daily_start = date_time.replace(hour=8)
    daily_start = daily_start.replace(minute=55)
    daily_start = daily_start.replace(second=0)
    daily_start = daily_start - datetime.timedelta(days=delta_days)
    return daily_start


def to_daily_end(date_time):
    daily_end = date_time + datetime.timedelta(days=1)
    daily_end = daily_end.replace(hour=8)
    daily_end = daily_end.replace(minute=55)
    daily_end = daily_end.replace(second=0)

    return daily_end


def prepare_data(item_name):
    global tokens_list, prices_list, trade_time_list, price_bins

    tokens_list = []
    prices_list = []
    trade_time_list = []
    print("prepare data..")
    csv_file_val = '/INPUT/lwb/k-emebedding/NORMAL/hybrid_data_val/csvs/' + item_name + '.csv'

    vol_chg_bins = vols_dict[item_name]

    # 判断是否为文件
    if os.path.isfile(csv_file_val):
        df = pd.read_csv(csv_file_val)
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        start_time = df.iloc[0]['trade_time']
        end_time = df.iloc[-1]['trade_time']
        current_time = start_time
        val_datas_30_5 = []
        while current_time <= end_time:
            # 5min
            start = str(to_daily_start(current_time))
            end = str(to_daily_end(current_time))
            frame_5min = df.query('trade_time >= @start and trade_time <= @end')[
                ['trade_time', 'open', 'close', 'high', 'low', 'vol', 'pre_close']]
            sen_5min = ["<5min>"]
            if frame_5min.empty:
                current_time = current_time + datetime.timedelta(days=1)
                continue
            else:
                frame_5min.sort_values('trade_time', inplace=True)
                frame_5min = frame_5min.reset_index(drop=True)
                frame_5min = frame_5min[frame_5min['vol'] >= 0]
                if frame_5min.shape[0] == 0:
                    current_time = current_time + datetime.timedelta(days=1)
                    continue
                else:
                    pass
                # price
                frame_5min['diff_open'] = (frame_5min['open'] - frame_5min['pre_close']) / frame_5min['pre_close']
                frame_5min['diff_close'] = (frame_5min['close'] - frame_5min['pre_close']) / frame_5min['pre_close']
                frame_5min['diff_high'] = (frame_5min['high'] - frame_5min['pre_close']) / frame_5min['pre_close']
                frame_5min['diff_low'] = (frame_5min['low'] - frame_5min['pre_close']) / frame_5min['pre_close']
                # norm bins
                v_max = norm_max_H5[item_name]
                price_bins_5min = [item / v_max for item in price_bins]
                ## norm diff_price
                frame_5min['open_dcr'] = pd.cut(frame_5min['diff_open'] / v_max, bins=price_bins_5min, labels=labels)
                frame_5min['close_dcr'] = pd.cut(frame_5min['diff_close'] / v_max, bins=price_bins_5min, labels=labels)
                frame_5min['high_dcr'] = pd.cut(frame_5min['diff_high'] / v_max, bins=price_bins_5min, labels=labels)
                frame_5min['low_dcr'] = pd.cut(frame_5min['diff_low'] / v_max, bins=price_bins_5min, labels=labels)
                # vol
                frame_5min['vol_dcr'] = pd.cut(frame_5min['vol'], bins=vol_chg_bins, labels=vol_labels)
                frame_5min['kbar_id'] = frame_5min['open_dcr'].astype('str') + frame_5min['close_dcr'].astype('str') + \
                                        frame_5min[
                                            'high_dcr'].astype('str') + frame_5min['low_dcr'].astype('str') + \
                                        frame_5min['vol_dcr'].astype('str')
                sen_5min += list(frame_5min['kbar_id'])
                opens = list(frame_5min["open"])
                closes = list(frame_5min["close"])
                highs = list(frame_5min["high"])
                lows = list(frame_5min["low"])
                prices = [[opens[i], closes[i], highs[i], lows[i]] for i in range(len(opens))]
                prices_list.append(prices)
                if tick_name == "D":
                    times = list(frame_5min['trade_date'])
                else:
                    times = list(frame_5min['trade_time'])[0]
                trade_time_list.append(times)

            # 15min
            last_daily = get_last_daily_time(current_time)
            start = to_daily_start(last_daily, 2)
            end = to_daily_end(last_daily)
            frame_15min = df.query('trade_time >= @start and trade_time <= @end')[
                ['trade_time', 'open_15min', 'close_15min', 'high_15min', 'low_15min', 'vol_15min', 'pre_close_15min']]
            frame_15min.dropna(inplace=True)
            sen_15min = ["<15min>"]
            if frame_15min.empty:
                pass
            else:
                # price
                frame_15min['diff_open'] = (frame_15min['open_15min'] - frame_15min['pre_close_15min']) / frame_15min[
                    'pre_close_15min']
                frame_15min['diff_close'] = (frame_15min['close_15min'] - frame_15min['pre_close_15min']) / frame_15min[
                    'pre_close_15min']
                frame_15min['diff_high'] = (frame_15min['high_15min'] - frame_15min['pre_close_15min']) / frame_15min[
                    'pre_close_15min']
                frame_15min['diff_low'] = (frame_15min['low_15min'] - frame_15min['pre_close_15min']) / frame_15min[
                    'pre_close_15min']
                # norm bins
                v_max = norm_max_H15[item_name]

                price_bins_15min = [item / v_max for item in price_bins]
                ## norm diff_price
                frame_15min['open_dcr'] = pd.cut(frame_15min['diff_open'] / v_max, bins=price_bins_15min, labels=labels)
                frame_15min['close_dcr'] = pd.cut(frame_15min['diff_close'] / v_max, bins=price_bins_15min,
                                                  labels=labels)
                frame_15min['high_dcr'] = pd.cut(frame_15min['diff_high'] / v_max, bins=price_bins_15min, labels=labels)
                frame_15min['low_dcr'] = pd.cut(frame_15min['diff_low'] / v_max, bins=price_bins_15min, labels=labels)
                # vol
                frame_15min['vol_dcr'] = pd.cut(frame_15min['vol_15min'], bins=vol_chg_bins, labels=vol_labels)
                frame_15min['kbar_id'] = frame_15min['open_dcr'].astype('str') + frame_15min['close_dcr'].astype(
                    'str') + \
                                         frame_15min[
                                             'high_dcr'].astype('str') + frame_15min['low_dcr'].astype('str') + \
                                         frame_15min[
                                             'vol_dcr'].astype('str')
                sen_15min += list(frame_15min['kbar_id'])
            # 30min
            start = to_daily_start(last_daily, 7)
            end = to_daily_end(last_daily)
            frame_30min = df.query('trade_time >= @start and trade_time <= @end')[
                ['trade_time', 'open_30min', 'close_30min', 'high_30min', 'low_30min', 'vol_30min', 'pre_close_30min']]
            frame_30min.dropna(inplace=True)
            sen_30min = ["<30min>"]
            if frame_30min.empty:
                pass
            else:
                # price
                frame_30min['diff_open'] = (frame_30min['open_30min'] - frame_30min['pre_close_30min']) / frame_30min[
                    'pre_close_30min']
                frame_30min['diff_close'] = (frame_30min['close_30min'] - frame_30min['pre_close_30min']) / frame_30min[
                    'pre_close_30min']
                frame_30min['diff_high'] = (frame_30min['high_30min'] - frame_30min['pre_close_30min']) / frame_30min[
                    'pre_close_30min']
                frame_30min['diff_low'] = (frame_30min['low_30min'] - frame_30min['pre_close_30min']) / frame_30min[
                    'pre_close_30min']
                # norm bins
                v_max = norm_max_H30[item_name]

                price_bins_30min = [item / v_max for item in price_bins]
                ## norm diff_price
                frame_30min['open_dcr'] = pd.cut(frame_30min['diff_open'] / v_max, bins=price_bins_30min, labels=labels)
                frame_30min['close_dcr'] = pd.cut(frame_30min['diff_close'] / v_max, bins=price_bins_30min,
                                                  labels=labels)
                frame_30min['high_dcr'] = pd.cut(frame_30min['diff_high'] / v_max, bins=price_bins_30min, labels=labels)
                frame_30min['low_dcr'] = pd.cut(frame_30min['diff_low'] / v_max, bins=price_bins_30min, labels=labels)
                # vol
                frame_30min['vol_dcr'] = pd.cut(frame_30min['vol_30min'], bins=vol_chg_bins, labels=vol_labels)
                frame_30min['kbar_id'] = frame_30min['open_dcr'].astype('str') + frame_30min['close_dcr'].astype(
                    'str') + \
                                         frame_30min[
                                             'high_dcr'].astype('str') + frame_30min['low_dcr'].astype('str') + \
                                         frame_30min[
                                             'vol_dcr'].astype('str')
                sen_30min += list(frame_30min['kbar_id'])
            # 60min
            start = to_daily_start(last_daily, 14)
            end = to_daily_end(last_daily)
            frame_60min = df.query('trade_time >= @start and trade_time <= @end')[
                ['trade_time', 'open_60min', 'close_60min', 'high_60min', 'low_60min', 'vol_60min', 'pre_close_60min']]
            frame_60min.dropna(inplace=True)
            sen_60min = ["<60min>"]
            if frame_60min.empty:
                pass
            else:
                # price
                frame_60min['diff_open'] = (frame_60min['open_60min'] - frame_60min['pre_close_60min']) / frame_60min[
                    'pre_close_60min']
                frame_60min['diff_close'] = (frame_60min['close_60min'] - frame_60min['pre_close_60min']) / frame_60min[
                    'pre_close_60min']
                frame_60min['diff_high'] = (frame_60min['high_60min'] - frame_60min['pre_close_60min']) / frame_60min[
                    'pre_close_60min']
                frame_60min['diff_low'] = (frame_60min['low_60min'] - frame_60min['pre_close_60min']) / frame_60min[
                    'pre_close_60min']
                # norm bins
                v_max = norm_max_H60[item_name]

                price_bins_60min = [item / v_max for item in price_bins]
                ## norm diff_price
                frame_60min['open_dcr'] = pd.cut(frame_60min['diff_open'] / v_max, bins=price_bins_60min, labels=labels)
                frame_60min['close_dcr'] = pd.cut(frame_60min['diff_close'] / v_max, bins=price_bins_60min,
                                                  labels=labels)
                frame_60min['high_dcr'] = pd.cut(frame_60min['diff_high'] / v_max, bins=price_bins_60min, labels=labels)
                frame_60min['low_dcr'] = pd.cut(frame_60min['diff_low'] / v_max, bins=price_bins_60min, labels=labels)
                # vol
                frame_60min['vol_dcr'] = pd.cut(frame_60min['vol_60min'], bins=vol_chg_bins, labels=vol_labels)
                frame_60min['kbar_id'] = frame_60min['open_dcr'].astype('str') + frame_60min['close_dcr'].astype(
                    'str') + \
                                         frame_60min[
                                             'high_dcr'].astype('str') + frame_60min['low_dcr'].astype('str') + \
                                         frame_60min[
                                             'vol_dcr'].astype('str')
                sen_60min += list(frame_60min['kbar_id'])
            # daily
            start = to_daily_start(last_daily, 45)
            end = to_daily_end(last_daily)
            frame_daily = df.query('trade_time >= @start and trade_time <= @end')[
                ['trade_time', 'open_daily', 'close_daily', 'high_daily', 'low_daily', 'vol_daily', 'pre_close_daily']]
            frame_daily.dropna(inplace=True)
            sen_daily = ["<daily>"]
            if frame_daily.empty:
                pass
            else:
                # price
                frame_daily['diff_open'] = (frame_daily['open_daily'] - frame_daily['pre_close_daily']) / frame_daily[
                    'pre_close_daily']
                frame_daily['diff_close'] = (frame_daily['close_daily'] - frame_daily['pre_close_daily']) / frame_daily[
                    'pre_close_daily']
                frame_daily['diff_high'] = (frame_daily['high_daily'] - frame_daily['pre_close_daily']) / frame_daily[
                    'pre_close_daily']
                frame_daily['diff_low'] = (frame_daily['low_daily'] - frame_daily['pre_close_daily']) / frame_daily[
                    'pre_close_daily']
                # norm bins
                v_max = norm_max_daily[item_name]

                price_bins_daily = [item / v_max for item in price_bins]
                ## norm diff_price
                frame_daily['open_dcr'] = pd.cut(frame_daily['diff_open'] / v_max, bins=price_bins_daily, labels=labels)
                frame_daily['close_dcr'] = pd.cut(frame_daily['diff_close'] / v_max, bins=price_bins_daily,
                                                  labels=labels)
                frame_daily['high_dcr'] = pd.cut(frame_daily['diff_high'] / v_max, bins=price_bins_daily, labels=labels)
                frame_daily['low_dcr'] = pd.cut(frame_daily['diff_low'] / v_max, bins=price_bins_daily, labels=labels)
                # vol
                frame_daily['vol_dcr'] = pd.cut(frame_daily['vol_daily'], bins=vol_chg_bins, labels=vol_labels)
                frame_daily['kbar_id'] = frame_daily['open_dcr'].astype('str') + frame_daily['close_dcr'].astype(
                    'str') + \
                                         frame_daily[
                                             'high_dcr'].astype('str') + frame_daily['low_dcr'].astype('str') + \
                                         frame_daily[
                                             'vol_dcr'].astype('str')
                sen_daily += list(frame_daily['kbar_id'])

            tokens_list.append(sen_30min + sen_5min)
            current_time = current_time + datetime.timedelta(days=1)
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


def get_a_signal(input_tokens, pre_signal=None, top_k=5, threshold=1.0, check_next=None):
    # print(input_tokens)
    input = torch.tensor(tokenizer.convert_tokens_to_ids(input_tokens)).unsqueeze(dim=0)

    # 预测一个bar
    with torch.no_grad():
        outputs = model(input)
        predictions = outputs[0]
    # 获取下一个单词的预测概率分布
    next_token_predictions = predictions[0, -1, :]  # * word_weights_list
    # print(next_token_predictions, len(next_token_predictions), max(next_token_predictions))
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
        # print(score, sum(score))
        signal = "sell"
    elif rise_score > threshold:
        # print(score, sum(score))
        signal = "buy"
    else:
        pass
    # true_label = cls_labels_dict[check_next]
    # if rise_score > 0 and true_label == 1 or rise_score < 0 and true_label == 0 or rise_score == 0 or check_next[1] == '0':
    #     ans = 1
    # else :
    #     print(check_next)
    #     ans = 0
    # if is_valid:
    #     if labels.index(pred_close) < labels.index('0'):
    #         signal = "sell"
    #     else:
    #         signal = "buy"
    # else:
    #     pass
    return signal


def get_signals(data_ids, top_k=5, threshold=1.0, st=15):
    global total_ans, pos_ans
    signals = [""] * len(data_ids)
    base_st = data_ids.index('<5min>')
    if base_st == 1:
        return signals, base_st
    for i in range(base_st + st, len(data_ids) - 1):
        input_tokens = data_ids[:i]
        signal = get_a_signal(input_tokens, pre_signal=signals[i - 2], top_k=top_k, threshold=threshold,
                              check_next=data_ids[i])
        signals[i - 1] = signal
    return signals, base_st


def compute_an_earning(prices, signals, fees_ratio, t_name, base_idx):
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
            idx = i - base_idx - 1
            earning += prices[idx + 1][1] - prices[idx][1]
            if prices[idx + 1][1] - prices[idx][1] >= 0:
                pos_count += 1
            # print(signals[i], ": ", prices[i + 1][1] - prices[i][1])
        elif signals[i] == "sell":
            idx = i - base_idx - 1
            earning += prices[idx][1] - prices[idx + 1][1]
            if prices[idx][1] - prices[idx + 1][1] >= 0:
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


def compute_earnings(data_ids_list, t_name, fees_ratio=0.0, top_k=5, threshold=1.0, st=15):
    earnings = []
    for i in range(len(data_ids_list)):
        signals, base_idx = get_signals(data_ids_list[i], top_k, threshold, st)
        earning, fees = compute_an_earning(prices_list[i], signals, fees_ratio, t_name, base_idx)
        if earning < 0:
            print("return: ", round(earning, 3), "trans fees:", round(fees, 3), " time:", trade_time_list[i])
        else:
            print("return: ", round(earning, 3), "trans fees:", round(fees, 3))
        earnings.append(earning)
    return earnings


if __name__ == "__main__":
    path = '/INPUT/lwb/k-emebedding/NORMAL/hybrid_data_val/csvs'
    t_fee_dict = {"RU": 0.3, "M": 0.3}
    # 获取目录下的所有子目录名
    item_names = [file.split('.')[-2] for file in os.listdir(path)]
    saved_names = list(vols_dict.keys())
    item_names = list(set(item_names).intersection(set(saved_names)))
    print(item_names)
    for item_name in item_names:
        print("=====================   ", item_name, "   =====================")
        prepare_data(item_name)
        # print(tokens_list[:5])
        png_save_path = "/home/lwb/Kbar/pngs/"
        position_ratio = 3
        base = prices_list[0][0][0] * position_ratio
        levearage = 10
        fees_ratio = 0.0000 + 0.000000
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
                                       fees_ratio=fees_ratio,
                                       top_k=5,
                                       threshold=0.0,
                                       st=15)
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
