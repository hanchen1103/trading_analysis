import os
import pickle
import pandas as pd
from typing import List
from algorithm.common import *
from back_test.BollingerBands import TradingRecord, BacktestRequest, get_bolling_backtest
from foundation.Logger import logger, init_logger
from ta.Technical import calculate_all_technical_indicator
from util.time import convert_str_to_unix_time_mil


def create_bollinger_bands_feature(symbol, interval, length, std_dev, start_day, end_day):
    init_logger()
    filename = f"feature_{symbol}_{interval}_{length}_{std_dev}_{start_day}_{end_day}.pkl"
    dir_path = "../static/cache"
    filepath = os.path.join(dir_path, filename)

    if os.path.exists(filepath):
        logger.info(f"Loading existing feature file: {filepath}")
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    df = calculate_all_technical_indicator(symbol, interval, start_day, end_day)
    req = BacktestRequest(
        symbol=symbol,
        interval=interval,
        length=length,
        stdDev=std_dev,
        startDay=start_day,
        endDay=end_day,
        volume=3000
    )
    trading_records = get_bolling_backtest(req).tradingRecords
    df = create_moving_window(df)
    df.fillna(method='bfill', inplace=True)
    feature_label_map = create_bollinger_bands_labels(trading_records, df)
    print(feature_label_map)
    with open(filepath, 'wb') as f:
        pickle.dump(feature_label_map, f)

    logger.info(f"Feature file created and saved to: {filepath}")
    return feature_label_map


def create_bollinger_bands_labels(trading_records: List[TradingRecord], k_lines: pd.DataFrame):
    labeled_sequences = []

    for record in trading_records:
        # 找到startTime和endTime对应的K线数据
        break_idx = k_lines[k_lines['close_time'] == convert_str_to_unix_time_mil(record.startTime)].index[0]

        start_idx = max(break_idx - 6 * 12, 0)

        # 提取该时间段内的K线序列
        sequence = k_lines.iloc[start_idx:break_idx + 1]

        # 确定标签：如果isFakeBreak为True，则标签为'假突破'，否则为'真突破'
        label = FAKE_BREAK_START_LABEL if record.isFakeBreak else TRUE_BREAK_START_LABEL

        # 创建一个包含序列和标签的字典，并添加到结果列表中
        labeled_sequences.append({"sequence": sequence, "label": label})

    return labeled_sequences


def create_moving_window(data_frame):
    window_size = 36

    cp = data_frame['close_price']
    hp = data_frame['highest_price']
    lp = data_frame['lowest_price']
    v = data_frame['volume']

    data_frame['rolling_close_price_mean'] = cp.rolling(window=window_size).mean()
    data_frame['rolling_close_price_max'] = cp.rolling(window=window_size).max()
    data_frame['rolling_close_price_min'] = cp.rolling(window=window_size).min()
    data_frame['rolling_close_price_std'] = cp.rolling(window=window_size).std()

    data_frame['rolling_highest_price_mean'] = hp.rolling(window=window_size).mean()
    data_frame['rolling_highest_price_max'] = hp.rolling(window=window_size).max()
    data_frame['rolling_highest_price_min'] = hp.rolling(window=window_size).min()
    data_frame['rolling_highest_price_std'] = hp.rolling(window=window_size).std()

    data_frame['rolling_lowest_price_mean'] = lp.rolling(window=window_size).mean()
    data_frame['rolling_lowest_price_max'] = lp.rolling(window=window_size).max()
    data_frame['rolling_lowest_price_min'] = lp.rolling(window=window_size).min()
    data_frame['rolling_lowest_price_std'] = lp.rolling(window=window_size).std()

    data_frame['rolling_volume_mean'] = v.rolling(window=window_size).mean()
    data_frame['rolling_volume_max'] = v.rolling(window=window_size).max()
    data_frame['rolling_volume_min'] = v.rolling(window=window_size).min()
    data_frame['rolling_volume_std'] = v.rolling(window=window_size).std()

    return data_frame


m = create_bollinger_bands_feature('perpusdt', '5m', 290, 0.6, 32, 0)
sequence_length = len(m)  # 获取序列长度
print(sequence_length)
feature_count = m[0]['sequence'].shape[1] - 4
print(feature_count)
