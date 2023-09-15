import os
import pandas as pd
from typing import List
from algorithm.common import *
from back_test.BollingerBands import TradingRecord, BacktestRequest, get_bolling_backtest
from foundation.Logger import logger, init_logger
from ta.Technical import calculate_all_technical_indicator
from util.time import convert_str_to_unix_time_mil


def create_bollinger_bands_feature(symbol, interval, length, std_dev, start_day, end_day):
    init_logger()
    filename = f"feature_{symbol}_{interval}_{length}_{std_dev}_{start_day}_{end_day}.csv"
    dir_path = "../static/cache"
    filepath = os.path.join(dir_path, filename)

    if os.path.exists(filepath):
        logger.info(f"Loading existing feature file: {filepath}")
        return pd.read_csv(filepath)

    logger.info(f"Creating new feature file: {filepath}")

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
    df = create_bollinger_bands_labels(trading_records, df)
    df = create_moving_window(df)
    df.fillna(method='bfill', inplace=True)

    df.to_csv(filepath, index=False)
    logger.info(f"Feature file created and saved to: {filepath}")
    return df


def create_bollinger_bands_labels(trading_records: List[TradingRecord], k_lines: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Create a new column for labels and initialize it with NO_BREAK_LABEL
    k_lines[BREAK_LABEL] = NO_BREAK_LABEL
    k_lines[TAKE_PROFIT_LABEL] = 0
    # Step 2 & 3: Set the correct labels based on the trading records
    for record in trading_records:

        start_time = convert_str_to_unix_time_mil(record.startTime)
        end_time = convert_str_to_unix_time_mil(record.endTime)
        max_profit_time = convert_str_to_unix_time_mil(record.maxProfitTime)

        start_index = k_lines[k_lines['close_time'] == start_time].index[0]
        end_index = k_lines[k_lines['close_time'] == end_time].index[0]
        max_profit_index = k_lines[k_lines['close_time'] == max_profit_time].index[0]

        # Set the label based on the record properties
        if record.isFakeBreak:
            k_lines.loc[start_index, BREAK_LABEL] = FAKE_BREAK_START_LABEL
            k_lines.loc[start_index + 1:end_index - 1, BREAK_LABEL] = FAKE_BREAK_MID_LABEL
            k_lines.loc[end_index, BREAK_LABEL] = FAKE_BREAK_END_LABEL
        else:
            k_lines.loc[start_index, BREAK_LABEL] = TRUE_BREAK_START_LABEL
            k_lines.loc[start_index + 1:end_index - 1, BREAK_LABEL] = TRUE_BREAK_MID_LABEL
            k_lines.loc[end_index, BREAK_LABEL] = TRUE_BREAK_END_LABEL

        k_lines.loc[max_profit_index, TAKE_PROFIT_LABEL] = 1
    logger.info('mark label success')
    return k_lines


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
