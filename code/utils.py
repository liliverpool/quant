import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo


def text_kbar(texts,
              pre_close_list,
              label_vol_dict,
              label_open_dict,
              label_close_dict,
              label_high_dict,
              label_low_dict,
              ):
    if isinstance(texts, str):
        texts = texts.split(' ')
    open_labels = [word[0] for word in texts]
    open_price_chg = [label_open_dict[label] for label in open_labels]
    open_prices = [x * y + y for x, y in zip(open_price_chg, pre_close_list)]

    close_labels = [word[1] for word in texts]
    close_price_chg = [label_close_dict[label] for label in close_labels]
    close_prices = [x * y + y for x, y in zip(close_price_chg, pre_close_list)]

    high_labels = [word[2] for word in texts]
    high_price_chg = [label_high_dict[label] for label in high_labels]
    high_prices = [x * y + y for x, y in zip(high_price_chg, pre_close_list)]

    low_labels = [word[3] for word in texts]
    low_price_chg = [label_low_dict[label] for label in low_labels]
    low_prices = [x * y + y for x, y in zip(low_price_chg, pre_close_list)]

    vol_labels = [word[4] for word in texts]
    vols = [label_vol_dict[label] for label in vol_labels]

    df = pd.DataFrame({'open': open_prices,
                       'close': close_prices,
                       'high': high_prices,
                       'low': low_prices,
                       'volume': vols})
    return df
    # mpf.plot(data, type='candle', mav=(), volume=True, savefig='test_kline_rec.png')


def plot_candlestick(df, file_path):
    # 创建一个Candlestick对象，传入价格数据
    candle = go.Candlestick(
        x=df.index,  # x轴为日期
        open=df['open'],  # 开盘价
        high=df['high'],  # 最高价
        low=df['low'],  # 最低价
        close=df['close']  # 收盘价
    )
    # 创建一个Bar对象，传入成交量数据
    volume = go.Bar(
        x=df.index,  # x轴为日期
        y=df['volume'],  # y轴为成交量
        yaxis='y2'  # 指定y轴为第二个坐标轴
    )
    # 创建一个Figure对象，添加Candlestick和Bar对象
    fig = go.Figure(data=[candle, volume])
    # 设置图表的布局，包括标题、坐标轴标签等
    fig.update_layout(
        title='K线图示例',
        xaxis_title='日期',
        yaxis_title='价格',
        # yaxis2_title='成交量',
        # yaxis2_showgrid=False,  # 不显示第二个y轴的网格线
        # xaxis_rangeslider_visible=False  # 不显示x轴的范围滑动条
    )
    fig.write_image(file_path)


def plot_kline(df, save_path):
    trace = go.Candlestick(x=df.index,
                           open=df['open'],
                           high=df['high'],
                           low=df['low'],
                           close=df['close'],
                           increasing=dict(line=dict(color='#00ff7f')),
                           decreasing=dict(line=dict(color='#ff6347')))
    data = [trace]
    layout = go.Layout(title='K线图', xaxis=dict(title='日期'), yaxis=dict(title='价格'))
    fig = go.Figure(data=data, layout=layout)
    pyo.plot(fig, filename=save_path, auto_open=False)


def load_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def gini_coefficient(freq_dict):
    if isinstance(freq_dict, dict):
        n = sum(freq_dict.values())
        freq_list = list(freq_dict.values())
    else:
        n = sum(freq_dict)
        freq_list = freq_dict
    if n == 0:
        return 0
    freq_list.sort()
    cum_freq = [sum(freq_list[:i + 1]) for i in range(len(freq_list))]
    return 1 - 2 * sum([f * cf for f, cf in zip(freq_list, cum_freq)]) / (n * sum(freq_list))


def count_words(lines):
    words = []
    for line in lines:
        line = line.translate(str.maketrans('', '', string.punctuation))  # 去除标点符号
        words += line.lower().split()
    return Counter(words)
