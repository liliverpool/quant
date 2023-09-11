import datetime
import json
import os
from argparse import ArgumentParser
from itertools import chain

import constants
import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup

# 定义评估指标
# from datasets import load_metric
# metric = load_metric("accuracy")

item_name = 'M'
tick_name = '5min'
vol_labels = constants.vol_labels_11
# bin 数据加载
hist_path = "/INPUT/lwb/k-emebedding/NORMAL/bin_intervals/"
with open(hist_path + "vol_bins_H5.json", "r") as f:
    vols_dict_H5 = json.load(f)
with open(hist_path + "vol_bins_H15.json", "r") as f:
    vols_dict_H15 = json.load(f)
with open(hist_path + "vol_bins_H30.json", "r") as f:
    vols_dict_H30 = json.load(f)
with open(hist_path + "vol_bins_H60.json", "r") as f:
    vols_dict_H60 = json.load(f)
with open(hist_path + "vol_bins_daily.json", "r") as f:
    vols_dict_daily = json.load(f)

# price bin
labels = constants.labels_13
price_bins = constants.price_diff_bins_13

parser = ArgumentParser()
# parser.add_argument("--save_path", type=str,
#                     default="/INPUT/lwb/k-emebedding/NORMAL/H-5_lseq-gpt-base-kbar-bin13-SIM")
parser.add_argument("--save_path", type=str,
                    default="/INPUT/lwb/k-emebedding/NORMAL/CLS/cls_model")
parser.add_argument("--save_cls_path", type=str,
                    default="/INPUT/lwb/k-emebedding/NORMAL/CLS/")
parser.add_argument("--pretrain_cls_labels_path", type=str,
                    default="/INPUT/lwb/k-emebedding/NORMAL/gpt2_tokenizer_bin13_H5/ft_cls_labels.json")
parser.add_argument("--tokenizer_path", type=str,
                    default="/INPUT/lwb/k-emebedding/NORMAL/gpt2_tokenizer_bin13_H5")
parser.add_argument("--train_path", type=str,
                    default='/INPUT/lwb/k-emebedding/NORMAL/hybrid_data_train/csvs/')
parser.add_argument("--valid_path", type=str,
                    default='/INPUT/lwb/k-emebedding/NORMAL/hybrid_data_val/csvs/')
args = parser.parse_args()

use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
# 加载CLS的字典
with open(args.pretrain_cls_labels_path, "r", encoding='gb2312') as f:
    cls_labels_dict = json.load(f)
# 数据集
tokens_list_H5_train = []
tokens_H5_train = []
trade_time_list_train = []
cls_data_train = []
tokens_list_H5_valid = []
tokens_H5_valid = []
trade_time_list_valid = []
cls_data_valid = []


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


def prepare_data_train(item_name, datapath):
    global tokens_list_H5_train, trade_time_list_train, tokens_H5_train, price_bins, vol_chg_bins
    tokens_list_H5_train = []
    trade_time_list_train = []
    print("prepare training data..")
    csv_file_val = datapath + item_name + '.csv'
    # 判断是否为文件
    if os.path.isfile(csv_file_val):
        df = pd.read_csv(csv_file_val)
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        start_time = df.iloc[0]['trade_time']
        end_time = df.iloc[-1]['trade_time']
        current_time = start_time
        while current_time <= end_time:
            # 5min
            start = str(to_daily_start(current_time))
            end = str(to_daily_end(current_time))
            frame_5min = df.query('trade_time >= @start and trade_time <= @end')[
                ['trade_time', 'open', 'close', 'high', 'low', 'vol', 'pre_close']]
            sen_5min = []
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
                vol_chg_bins = vols_dict_H5[item_name]
                frame_5min['vol_dcr'] = pd.cut(frame_5min['vol'], bins=vol_chg_bins, labels=vol_labels)
                frame_5min['diff_open'] = (frame_5min['open'] - frame_5min['pre_close']) / frame_5min['pre_close']
                frame_5min['open_dcr'] = pd.cut(frame_5min['diff_open'], bins=price_bins, labels=labels)
                frame_5min['diff_close'] = (frame_5min['close'] - frame_5min['pre_close']) / frame_5min['pre_close']
                frame_5min['close_dcr'] = pd.cut(frame_5min['diff_close'], bins=price_bins, labels=labels)
                frame_5min['diff_high'] = (frame_5min['high'] - frame_5min['pre_close']) / frame_5min['pre_close']
                frame_5min['high_dcr'] = pd.cut(frame_5min['diff_high'], bins=price_bins, labels=labels)
                frame_5min['diff_low'] = (frame_5min['low'] - frame_5min['pre_close']) / frame_5min['pre_close']
                frame_5min['low_dcr'] = pd.cut(frame_5min['diff_low'], bins=price_bins, labels=labels)
                frame_5min['kbar_id'] = frame_5min['open_dcr'].astype('str') + frame_5min['close_dcr'].astype('str') + \
                                        frame_5min[
                                            'high_dcr'].astype('str') + frame_5min['low_dcr'].astype('str') + \
                                        frame_5min['vol_dcr'].astype('str')
                sen_5min += list(frame_5min['kbar_id'])
                if tick_name == "D":
                    times = list(frame_5min['trade_date'])
                else:
                    times = list(frame_5min['trade_time'])[0]
                trade_time_list_train.append(times)
                opens = list(frame_5min["open"])
                closes = list(frame_5min["close"])
                highs = list(frame_5min["high"])
                lows = list(frame_5min["low"])

            # 15min
            start = str(to_daily_start(current_time))
            end = str(to_daily_end(current_time))
            frame_15min = df.query('trade_time >= @start and trade_time <= @end')[
                ['trade_time', 'open_15min', 'close_15min', 'high_15min', 'low_15min', 'vol_15min', 'pre_close_15min']]
            frame_15min.dropna(inplace=True)
            sen_15min = []
            if frame_15min.empty:
                pass
            else:
                vol_chg_bins = vols_dict_H15[item_name]
                frame_15min['vol_dcr'] = pd.cut(frame_15min['vol_15min'], bins=vol_chg_bins, labels=vol_labels)
                frame_15min['diff_open'] = (frame_15min['open_15min'] - frame_15min['pre_close_15min']) / frame_15min[
                    'pre_close_15min']
                frame_15min['open_dcr'] = pd.cut(frame_15min['diff_open'], bins=price_bins, labels=labels)
                frame_15min['diff_close'] = (frame_15min['close_15min'] - frame_15min['pre_close_15min']) / frame_15min[
                    'pre_close_15min']
                frame_15min['close_dcr'] = pd.cut(frame_15min['diff_close'], bins=price_bins, labels=labels)
                frame_15min['diff_high'] = (frame_15min['high_15min'] - frame_15min['pre_close_15min']) / frame_15min[
                    'pre_close_15min']
                frame_15min['high_dcr'] = pd.cut(frame_15min['diff_high'], bins=price_bins, labels=labels)
                frame_15min['diff_low'] = (frame_15min['low_15min'] - frame_15min['pre_close_15min']) / frame_15min[
                    'pre_close_15min']
                frame_15min['low_dcr'] = pd.cut(frame_15min['diff_low'], bins=price_bins, labels=labels)
                frame_15min['kbar_id'] = frame_15min['open_dcr'].astype('str') + frame_15min['close_dcr'].astype(
                    'str') + \
                                         frame_15min[
                                             'high_dcr'].astype('str') + frame_15min['low_dcr'].astype('str') + \
                                         frame_15min[
                                             'vol_dcr'].astype('str')
                sen_15min += list(frame_15min['kbar_id'])

            # 30min
            start = str(to_daily_start(current_time))
            end = str(to_daily_end(current_time))
            frame_30min = df.query('trade_time >= @start and trade_time <= @end')[
                ['trade_time', 'open_30min', 'close_30min', 'high_30min', 'low_30min', 'vol_30min', 'pre_close_30min']]
            frame_30min.dropna(inplace=True)
            sen_30min = []
            if frame_30min.empty:
                pass
            else:
                vol_chg_bins = vols_dict_H30[item_name]
                frame_30min['vol_dcr'] = pd.cut(frame_30min['vol_30min'], bins=vol_chg_bins, labels=vol_labels)
                frame_30min['diff_open'] = (frame_30min['open_30min'] - frame_30min['pre_close_30min']) / frame_30min[
                    'pre_close_30min']
                frame_30min['open_dcr'] = pd.cut(frame_30min['diff_open'], bins=price_bins, labels=labels)
                frame_30min['diff_close'] = (frame_30min['close_30min'] - frame_30min['pre_close_30min']) / frame_30min[
                    'pre_close_30min']
                frame_30min['close_dcr'] = pd.cut(frame_30min['diff_close'], bins=price_bins, labels=labels)
                frame_30min['diff_high'] = (frame_30min['high_30min'] - frame_30min['pre_close_30min']) / frame_30min[
                    'pre_close_30min']
                frame_30min['high_dcr'] = pd.cut(frame_30min['diff_high'], bins=price_bins, labels=labels)
                frame_30min['diff_low'] = (frame_30min['low_30min'] - frame_30min['pre_close_30min']) / frame_30min[
                    'pre_close_30min']
                frame_30min['low_dcr'] = pd.cut(frame_30min['diff_low'], bins=price_bins, labels=labels)
                frame_30min['kbar_id'] = frame_30min['open_dcr'].astype('str') + frame_30min['close_dcr'].astype(
                    'str') + \
                                         frame_30min[
                                             'high_dcr'].astype('str') + frame_30min['low_dcr'].astype('str') + \
                                         frame_30min[
                                             'vol_dcr'].astype('str')
                sen_30min += list(frame_30min['kbar_id'])
            # 60min
            start = str(to_daily_start(current_time))
            end = str(to_daily_end(current_time))
            frame_60min = df.query('trade_time >= @start and trade_time <= @end')[
                ['trade_time', 'open_60min', 'close_60min', 'high_60min', 'low_60min', 'vol_60min', 'pre_close_60min']]
            frame_60min.dropna(inplace=True)
            sen_60min = []
            if frame_60min.empty:
                pass
            else:
                vol_chg_bins = vols_dict_H60[item_name]
                frame_60min['vol_dcr'] = pd.cut(frame_60min['vol_60min'], bins=vol_chg_bins, labels=vol_labels)
                frame_60min['diff_open'] = (frame_60min['open_60min'] - frame_60min['pre_close_60min']) / frame_60min[
                    'pre_close_60min']
                frame_60min['open_dcr'] = pd.cut(frame_60min['diff_open'], bins=price_bins, labels=labels)
                frame_60min['diff_close'] = (frame_60min['close_60min'] - frame_60min['pre_close_60min']) / frame_60min[
                    'pre_close_60min']
                frame_60min['close_dcr'] = pd.cut(frame_60min['diff_close'], bins=price_bins, labels=labels)
                frame_60min['diff_high'] = (frame_60min['high_60min'] - frame_60min['pre_close_60min']) / frame_60min[
                    'pre_close_60min']
                frame_60min['high_dcr'] = pd.cut(frame_60min['diff_high'], bins=price_bins, labels=labels)
                frame_60min['diff_low'] = (frame_60min['low_60min'] - frame_60min['pre_close_60min']) / frame_60min[
                    'pre_close_60min']
                frame_60min['low_dcr'] = pd.cut(frame_60min['diff_low'], bins=price_bins, labels=labels)
                frame_60min['kbar_id'] = frame_60min['open_dcr'].astype('str') + frame_60min['close_dcr'].astype(
                    'str') + \
                                         frame_60min[
                                             'high_dcr'].astype('str') + frame_60min['low_dcr'].astype('str') + \
                                         frame_60min[
                                             'vol_dcr'].astype('str')
                sen_60min += list(frame_60min['kbar_id'])
            # daily
            start = str(to_daily_start(current_time))
            end = str(to_daily_end(current_time))
            frame_daily = df.query('trade_time >= @start and trade_time <= @end')[
                ['trade_time', 'open_daily', 'close_daily', 'high_daily', 'low_daily', 'vol_daily', 'pre_close_daily']]
            frame_daily.dropna(inplace=True)
            sen_daily = []
            if frame_daily.empty:
                pass
            else:
                vol_chg_bins = vols_dict_daily[item_name]
                frame_daily['vol_dcr'] = pd.cut(frame_daily['vol_daily'], bins=vol_chg_bins, labels=vol_labels)
                frame_daily['diff_open'] = (frame_daily['open_daily'] - frame_daily['pre_close_daily']) / frame_daily[
                    'pre_close_daily']
                frame_daily['open_dcr'] = pd.cut(frame_daily['diff_open'], bins=price_bins, labels=labels)
                frame_daily['diff_close'] = (frame_daily['close_daily'] - frame_daily['pre_close_daily']) / frame_daily[
                    'pre_close_daily']
                frame_daily['close_dcr'] = pd.cut(frame_daily['diff_close'], bins=price_bins, labels=labels)
                frame_daily['diff_high'] = (frame_daily['high_daily'] - frame_daily['pre_close_daily']) / frame_daily[
                    'pre_close_daily']
                frame_daily['high_dcr'] = pd.cut(frame_daily['diff_high'], bins=price_bins, labels=labels)
                frame_daily['diff_low'] = (frame_daily['low_daily'] - frame_daily['pre_close_daily']) / frame_daily[
                    'pre_close_daily']
                frame_daily['low_dcr'] = pd.cut(frame_daily['diff_low'], bins=price_bins, labels=labels)
                frame_daily['kbar_id'] = frame_daily['open_dcr'].astype('str') + frame_daily['close_dcr'].astype(
                    'str') + \
                                         frame_daily[
                                             'high_dcr'].astype('str') + frame_daily['low_dcr'].astype('str') + \
                                         frame_daily[
                                             'vol_dcr'].astype('str')
                sen_daily += list(frame_daily['kbar_id'])

            tokens_list_H5_train.append(sen_5min)
            current_time = current_time + datetime.timedelta(days=1)
    zipped = list(zip(tokens_list_H5_train, trade_time_list_train))
    sorted_zipped = sorted(zipped, key=lambda x: x[-1])

    tokens_list_H5_train, trade_time_list_train = zip(*sorted_zipped)
    tokens_H5_train = list(chain.from_iterable(tokens_list_H5_train))


def prepare_cls_data_train(item_name, st=249):
    prepare_data_train(item_name, args.train_path)
    for i in range(st, len(tokens_H5_train)):
        cls_data_tokens = tokens_H5_train[i - st:i]
        cls_label = cls_labels_dict[cls_data_tokens[-1]] + 1
        if cls_label == 1:
            continue
        elif cls_label == 0:
            cls_data_train.append([cls_data_tokens[:st - 1], cls_label])
        elif cls_label == 2:
            cls_data_train.append([cls_data_tokens[:st - 1], 1])
        else:
            continue


def prepare_data_valid(item_name, datapath):
    global tokens_list_H5_valid, trade_time_list_valid, tokens_H5_valid, price_bins, vol_chg_bins
    tokens_list_H5_valid = []
    trade_time_list_valid = []
    print("prepare valid data..")
    csv_file_val = datapath + item_name + '.csv'

    # 判断是否为文件
    if os.path.isfile(csv_file_val):
        df = pd.read_csv(csv_file_val)
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        start_time = df.iloc[0]['trade_time']
        end_time = df.iloc[-1]['trade_time']
        current_time = start_time
        while current_time <= end_time:
            # 5min
            start = str(to_daily_start(current_time))
            end = str(to_daily_end(current_time))
            frame_5min = df.query('trade_time >= @start and trade_time <= @end')[
                ['trade_time', 'open', 'close', 'high', 'low', 'vol', 'pre_close']]
            sen_5min = []
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
                vol_chg_bins = vols_dict_H5[item_name]
                frame_5min['vol_dcr'] = pd.cut(frame_5min['vol'], bins=vol_chg_bins, labels=vol_labels)
                frame_5min['diff_open'] = (frame_5min['open'] - frame_5min['pre_close']) / frame_5min['pre_close']
                frame_5min['open_dcr'] = pd.cut(frame_5min['diff_open'], bins=price_bins, labels=labels)
                frame_5min['diff_close'] = (frame_5min['close'] - frame_5min['pre_close']) / frame_5min['pre_close']
                frame_5min['close_dcr'] = pd.cut(frame_5min['diff_close'], bins=price_bins, labels=labels)
                frame_5min['diff_high'] = (frame_5min['high'] - frame_5min['pre_close']) / frame_5min['pre_close']
                frame_5min['high_dcr'] = pd.cut(frame_5min['diff_high'], bins=price_bins, labels=labels)
                frame_5min['diff_low'] = (frame_5min['low'] - frame_5min['pre_close']) / frame_5min['pre_close']
                frame_5min['low_dcr'] = pd.cut(frame_5min['diff_low'], bins=price_bins, labels=labels)
                frame_5min['kbar_id'] = frame_5min['open_dcr'].astype('str') + frame_5min['close_dcr'].astype('str') + \
                                        frame_5min[
                                            'high_dcr'].astype('str') + frame_5min['low_dcr'].astype('str') + \
                                        frame_5min['vol_dcr'].astype('str')
                sen_5min += list(frame_5min['kbar_id'])
                if tick_name == "D":
                    times = list(frame_5min['trade_date'])
                else:
                    times = list(frame_5min['trade_time'])[0]
                trade_time_list_valid.append(times)
                opens = list(frame_5min["open"])
                closes = list(frame_5min["close"])
                highs = list(frame_5min["high"])
                lows = list(frame_5min["low"])

            # 15min
            start = str(to_daily_start(current_time))
            end = str(to_daily_end(current_time))
            frame_15min = df.query('trade_time >= @start and trade_time <= @end')[
                ['trade_time', 'open_15min', 'close_15min', 'high_15min', 'low_15min', 'vol_15min', 'pre_close_15min']]
            frame_15min.dropna(inplace=True)
            sen_15min = []
            if frame_15min.empty:
                pass
            else:
                vol_chg_bins = vols_dict_H15[item_name]
                frame_15min['vol_dcr'] = pd.cut(frame_15min['vol_15min'], bins=vol_chg_bins, labels=vol_labels)
                frame_15min['diff_open'] = (frame_15min['open_15min'] - frame_15min['pre_close_15min']) / frame_15min[
                    'pre_close_15min']
                frame_15min['open_dcr'] = pd.cut(frame_15min['diff_open'], bins=price_bins, labels=labels)
                frame_15min['diff_close'] = (frame_15min['close_15min'] - frame_15min['pre_close_15min']) / frame_15min[
                    'pre_close_15min']
                frame_15min['close_dcr'] = pd.cut(frame_15min['diff_close'], bins=price_bins, labels=labels)
                frame_15min['diff_high'] = (frame_15min['high_15min'] - frame_15min['pre_close_15min']) / frame_15min[
                    'pre_close_15min']
                frame_15min['high_dcr'] = pd.cut(frame_15min['diff_high'], bins=price_bins, labels=labels)
                frame_15min['diff_low'] = (frame_15min['low_15min'] - frame_15min['pre_close_15min']) / frame_15min[
                    'pre_close_15min']
                frame_15min['low_dcr'] = pd.cut(frame_15min['diff_low'], bins=price_bins, labels=labels)
                frame_15min['kbar_id'] = frame_15min['open_dcr'].astype('str') + frame_15min['close_dcr'].astype(
                    'str') + \
                                         frame_15min[
                                             'high_dcr'].astype('str') + frame_15min['low_dcr'].astype('str') + \
                                         frame_15min[
                                             'vol_dcr'].astype('str')
                sen_15min += list(frame_15min['kbar_id'])

            # 30min
            start = str(to_daily_start(current_time))
            end = str(to_daily_end(current_time))
            frame_30min = df.query('trade_time >= @start and trade_time <= @end')[
                ['trade_time', 'open_30min', 'close_30min', 'high_30min', 'low_30min', 'vol_30min', 'pre_close_30min']]
            frame_30min.dropna(inplace=True)
            sen_30min = []
            if frame_30min.empty:
                pass
            else:
                vol_chg_bins = vols_dict_H30[item_name]
                frame_30min['vol_dcr'] = pd.cut(frame_30min['vol_30min'], bins=vol_chg_bins, labels=vol_labels)
                frame_30min['diff_open'] = (frame_30min['open_30min'] - frame_30min['pre_close_30min']) / frame_30min[
                    'pre_close_30min']
                frame_30min['open_dcr'] = pd.cut(frame_30min['diff_open'], bins=price_bins, labels=labels)
                frame_30min['diff_close'] = (frame_30min['close_30min'] - frame_30min['pre_close_30min']) / frame_30min[
                    'pre_close_30min']
                frame_30min['close_dcr'] = pd.cut(frame_30min['diff_close'], bins=price_bins, labels=labels)
                frame_30min['diff_high'] = (frame_30min['high_30min'] - frame_30min['pre_close_30min']) / frame_30min[
                    'pre_close_30min']
                frame_30min['high_dcr'] = pd.cut(frame_30min['diff_high'], bins=price_bins, labels=labels)
                frame_30min['diff_low'] = (frame_30min['low_30min'] - frame_30min['pre_close_30min']) / frame_30min[
                    'pre_close_30min']
                frame_30min['low_dcr'] = pd.cut(frame_30min['diff_low'], bins=price_bins, labels=labels)
                frame_30min['kbar_id'] = frame_30min['open_dcr'].astype('str') + frame_30min['close_dcr'].astype(
                    'str') + \
                                         frame_30min[
                                             'high_dcr'].astype('str') + frame_30min['low_dcr'].astype('str') + \
                                         frame_30min[
                                             'vol_dcr'].astype('str')
                sen_30min += list(frame_30min['kbar_id'])
            # 60min
            start = str(to_daily_start(current_time))
            end = str(to_daily_end(current_time))
            frame_60min = df.query('trade_time >= @start and trade_time <= @end')[
                ['trade_time', 'open_60min', 'close_60min', 'high_60min', 'low_60min', 'vol_60min', 'pre_close_60min']]
            frame_60min.dropna(inplace=True)
            sen_60min = []
            if frame_60min.empty:
                pass
            else:
                vol_chg_bins = vols_dict_H60[item_name]
                frame_60min['vol_dcr'] = pd.cut(frame_60min['vol_60min'], bins=vol_chg_bins, labels=vol_labels)
                frame_60min['diff_open'] = (frame_60min['open_60min'] - frame_60min['pre_close_60min']) / frame_60min[
                    'pre_close_60min']
                frame_60min['open_dcr'] = pd.cut(frame_60min['diff_open'], bins=price_bins, labels=labels)
                frame_60min['diff_close'] = (frame_60min['close_60min'] - frame_60min['pre_close_60min']) / frame_60min[
                    'pre_close_60min']
                frame_60min['close_dcr'] = pd.cut(frame_60min['diff_close'], bins=price_bins, labels=labels)
                frame_60min['diff_high'] = (frame_60min['high_60min'] - frame_60min['pre_close_60min']) / frame_60min[
                    'pre_close_60min']
                frame_60min['high_dcr'] = pd.cut(frame_60min['diff_high'], bins=price_bins, labels=labels)
                frame_60min['diff_low'] = (frame_60min['low_60min'] - frame_60min['pre_close_60min']) / frame_60min[
                    'pre_close_60min']
                frame_60min['low_dcr'] = pd.cut(frame_60min['diff_low'], bins=price_bins, labels=labels)
                frame_60min['kbar_id'] = frame_60min['open_dcr'].astype('str') + frame_60min['close_dcr'].astype(
                    'str') + \
                                         frame_60min[
                                             'high_dcr'].astype('str') + frame_60min['low_dcr'].astype('str') + \
                                         frame_60min[
                                             'vol_dcr'].astype('str')
                sen_60min += list(frame_60min['kbar_id'])
            # daily
            start = str(to_daily_start(current_time))
            end = str(to_daily_end(current_time))
            frame_daily = df.query('trade_time >= @start and trade_time <= @end')[
                ['trade_time', 'open_daily', 'close_daily', 'high_daily', 'low_daily', 'vol_daily', 'pre_close_daily']]
            frame_daily.dropna(inplace=True)
            sen_daily = []
            if frame_daily.empty:
                pass
            else:
                vol_chg_bins = vols_dict_daily[item_name]
                frame_daily['vol_dcr'] = pd.cut(frame_daily['vol_daily'], bins=vol_chg_bins, labels=vol_labels)
                frame_daily['diff_open'] = (frame_daily['open_daily'] - frame_daily['pre_close_daily']) / frame_daily[
                    'pre_close_daily']
                frame_daily['open_dcr'] = pd.cut(frame_daily['diff_open'], bins=price_bins, labels=labels)
                frame_daily['diff_close'] = (frame_daily['close_daily'] - frame_daily['pre_close_daily']) / frame_daily[
                    'pre_close_daily']
                frame_daily['close_dcr'] = pd.cut(frame_daily['diff_close'], bins=price_bins, labels=labels)
                frame_daily['diff_high'] = (frame_daily['high_daily'] - frame_daily['pre_close_daily']) / frame_daily[
                    'pre_close_daily']
                frame_daily['high_dcr'] = pd.cut(frame_daily['diff_high'], bins=price_bins, labels=labels)
                frame_daily['diff_low'] = (frame_daily['low_daily'] - frame_daily['pre_close_daily']) / frame_daily[
                    'pre_close_daily']
                frame_daily['low_dcr'] = pd.cut(frame_daily['diff_low'], bins=price_bins, labels=labels)
                frame_daily['kbar_id'] = frame_daily['open_dcr'].astype('str') + frame_daily['close_dcr'].astype(
                    'str') + \
                                         frame_daily[
                                             'high_dcr'].astype('str') + frame_daily['low_dcr'].astype('str') + \
                                         frame_daily[
                                             'vol_dcr'].astype('str')
                sen_daily += list(frame_daily['kbar_id'])
            tokens_list_H5_valid.append(sen_5min)
            current_time = current_time + datetime.timedelta(days=1)
    zipped = list(zip(tokens_list_H5_valid, trade_time_list_valid))
    sorted_zipped = sorted(zipped, key=lambda x: x[-1])
    tokens_list_H5_valid, trade_time_list_valid = zip(*sorted_zipped)
    tokens_H5_valid = list(chain.from_iterable(tokens_list_H5_valid))


def prepare_cls_data_valid(item_name, st=249):
    prepare_data_valid(item_name, args.valid_path)
    for i in range(st, len(tokens_H5_valid)):
        cls_data_tokens = tokens_H5_valid[i - st:i]
        cls_label = cls_labels_dict[cls_data_tokens[-1]] + 1
        if cls_label == 1:
            continue
        elif cls_label == 0:
            cls_data_valid.append([cls_data_tokens[:st - 1], cls_label])
        elif cls_label == 2:
            cls_data_valid.append([cls_data_tokens[:st - 1], 1])
        else:
            continue


# 加载预训练模型和tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

prepare_cls_data_train(item_name, 248)

prepare_cls_data_valid(item_name, 248)

input_ids_train = []
labels_train = []
padding_token_id = 0
for item in cls_data_train:
    sent = item[0]
    label = item[1]
    input_ids_train.append(tokenizer.convert_tokens_to_ids(sent))
    labels_train.append(label)
# input_ids_train = pad_sequence(input_ids_train, batch_first=True, padding_value=padding_token_id)

input_ids_val = []
labels_val = []
for item in cls_data_valid:
    sent = item[0]
    label = item[1]
    input_ids_val.append(tokenizer.convert_tokens_to_ids(sent))
    labels_val.append(label)
# input_ids_val = pad_sequence(input_ids_val, batch_first=True, padding_value=padding_token_id)

labels_train = torch.tensor(labels_train).to(device)
labels_val = torch.tensor(labels_val).to(device)
input_ids_train = torch.tensor(input_ids_train).to(device)
input_ids_val = torch.tensor(input_ids_val).to(device)

# 将数据集划分为训练集和验证集：
from torch.utils.data import TensorDataset

train_dataset = TensorDataset(input_ids_train,
                              labels_train)
val_dataset = TensorDataset(input_ids_val,
                            labels_val)

# 定义训练参数：
batch_size = 64
learning_rate = 1e-5  # 假设每个GPU上使用0.001的学习率

print(len(train_dataset))
print(len(val_dataset))
train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)

validation_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=batch_size
)

classes = torch.unique(labels_train).tolist()

label_num_dict = {label: labels_train.tolist().count(label) for label in classes}
cls_num = len(label_num_dict.keys())
weights = [(1 - label_num_dict[label] / len(labels_train)) ** 3 * 10 for label in classes]
print(classes, weights)
print(label_num_dict)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
if cls_num == 2:
    task = "binary"
else:
    task = "multiclass"

model = GPT2ForSequenceClassification.from_pretrained(args.save_path, num_labels=cls_num)
model.config.pad_token_id = model.config.eos_token_id

optimizer = AdamW(model.parameters(),
                  lr=learning_rate,
                  eps=1e-8
                  )

epochs = 20
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

print("start training ...")
# 定义训练循环
max_score = 0
for epoch in range(epochs):
    nb_train_steps = 0
    train_accuracy = 0
    for step, batch in enumerate(
            tqdm(train_dataloader, desc="Epoch: {}/{}".format(epoch + 1, epochs), bar_format="{l_bar}{bar:30}{r_bar}")):
        model.train()
        batch_input_ids = batch[0].to(device)
        batch_labels = batch[1].to(device)
        model.zero_grad()
        model.to(device)
        outputs = model(batch_input_ids, labels=batch_labels)
        criterion.to(device)
        loss = criterion(outputs.logits.to(device), batch_labels)

        # loss = outputs.loss
        logits = outputs.logits
        logits = logits.detach()
        preds = torch.argmax(logits, dim=-1)  # 将logits转换为类别预测
        tmp_train_accuracy = torchmetrics.functional.accuracy(preds, batch_labels, average='macro',
                                                                 task=task, num_classes=cls_num)
        train_accuracy += tmp_train_accuracy
        nb_train_steps += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    all_eval_accuracy = 0
    all_nb_eval_steps, all_nb_eval_examples = 0, 0
    high_conf_count = 0
    total_count = 0
    label_batch_num_dict = {}
    # 定义类别标签

    for label in classes:
        label_batch_num_dict[label] = 0
    # 初始化类别准确度字典
    class_accuracy = {label: 0 for label in classes}
    label_num_dict = {label: 0 for label in classes}

    for batch in validation_dataloader:
        batch_input_ids = batch[0].to(device)
        batch_labels = batch[1].to(device)
        with torch.no_grad():
            outputs = model(batch_input_ids, labels=batch_labels)
            loss = outputs.loss
            logits = outputs.logits
            logits = logits.detach()
            probs = F.softmax(logits, dim=-1)
            confidences = probs.max(dim=-1).values
        eval_loss += loss.item()
        threshold = 0.7
        # 计算符合置信度要求的样本的准确度
        mask = confidences > threshold
        if mask.sum() > 0:

            for label in classes:
                msk = torch.logical_and(batch_labels == label, confidences > threshold)
                if msk.sum() > 0:
                    label_batch_num_dict[label] += 1

            high_conf_count += mask.sum().item()
            preds = torch.argmax(logits[mask], dim=-1)  # 将logits转换为类别预测
            tmp_eval_accuracy = torchmetrics.functional.accuracy(preds, batch_labels[mask], average='macro',
                                                                 task=task, num_classes=cls_num)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            nb_eval_examples += batch_labels.size(0)
        else:
            pass
        total_count += batch_labels.size(0)
        preds = torch.argmax(logits, dim=-1)  # 将logits转换为类别预测
        tmp_eval_accuracy = torchmetrics.functional.accuracy(preds, batch_labels, task=task, average='macro',
                                                             num_classes=cls_num)
        all_eval_accuracy += tmp_eval_accuracy
        all_nb_eval_steps += 1
        all_nb_eval_examples += batch_labels.size(0)

        # 计算各个类别的准确度
        for label in classes:
            mask = torch.logical_and(batch_labels == label, confidences > threshold)
            label_num_dict[label] += mask.sum().item()
            if mask.sum() > 0:
                preds = torch.argmax(logits[mask], dim=-1)  # 将logits转换为类别预测
                tmp_class_accuracy = torchmetrics.functional.accuracy(preds, batch_labels[mask], average='macro',
                                                                      task=task, num_classes=cls_num)
                class_accuracy[label] += tmp_class_accuracy

    # print()
    print("Epoch: {}/{}".format(epoch + 1, epochs))
    print("Training Loss: {:.3f}".format(loss.item()))
    print("Training Accuracy: {:.3f}".format(train_accuracy / nb_train_steps))
    if nb_eval_steps > 0:
        print(high_conf_count, total_count, "high confidence ratio:", high_conf_count / total_count)
        print("Validation Accuracy: {:.3f}".format(eval_accuracy / nb_eval_steps))
    else:
        print("--no high confidence--")
    print("Total Validation Accuracy: {:.3f}".format(all_eval_accuracy / all_nb_eval_steps))

    # 显示各个类别的准确度
    for label in classes:
        if label_batch_num_dict[label] == 0:
            continue
        print(label, " num (training):", label_num_dict[label])
        print("Class {} Accuracy: {:.3f}".format(label, class_accuracy[label] / label_batch_num_dict[label]))
    torch.save(model.state_dict(), args.save_cls_path + 'cls_' + tick_name + '_gpt_model_cls_' + item_name + "_" + str(
        eval_accuracy / max(nb_eval_steps, 1)) + '.bin')
# 保存模型：
torch.save(model.state_dict(),  args.save_cls_path + tick_name + '_gpt_model_cls_' + item_name + '.bin')
