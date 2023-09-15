import numpy as np
import pandas as pd
import talib

from sklearn.cluster import KMeans

from dao.KLineDataDao import get_kline_data
from foundation.Logger import logger
from model.common import interval_to_data_points
from ta.common import *
from concurrent.futures import ThreadPoolExecutor

technical_executor = ThreadPoolExecutor(max_workers=9)


def convert_k_lines_to_df(k_lines):
    data = {
        'id': [],
        'open_price': [],
        'highest_price': [],
        'lowest_price': [],
        'close_price': [],
        'volume': [],
        'close_time': [],
        'transaction_amount': [],
        'transaction_count': [],
        'active_buy_volume': [],
        'active_buy_amount': [],
    }

    for k in k_lines:
        data['id'].append(k.id)
        data['open_price'].append(k.open_price)
        data['highest_price'].append(k.highest_price)
        data['lowest_price'].append(k.lowest_price)
        data['close_price'].append(k.close_price)
        data['volume'].append(k.volume)
        data['close_time'].append(k.close_time)
        data['transaction_amount'].append(k.transaction_amount)
        data['transaction_count'].append(k.transaction_count)
        data['active_buy_volume'].append(k.active_buy_volume)
        data['active_buy_amount'].append(k.active_buy_amount)

    return pd.DataFrame(data)


def get_bolling_bands(d, length=20, std_dev=2.0):
    d[BOLLING_UPPER], d[BOLLING_MIDDLE], d[BOLLING_LOWER] = talib.BBANDS(d['close_price'].astype(np.float64),
                                                                         timeperiod=length,
                                                                         nbdevup=std_dev,
                                                                         nbdevdn=std_dev, matype=0)
    logger.info('calculate bolling bands success')
    return d


def get_bolling_bands_b(d, length=290, std_dev=0.6):
    d[BOLLING_UPPER_B], d[BOLLING_MIDDLE_B], d[BOLLING_LOWER_B] = talib.BBANDS(d['close_price'].astype(np.float64),
                                                                               timeperiod=length,
                                                                               nbdevup=std_dev,
                                                                               nbdevdn=std_dev, matype=0)
    logger.info('calculate bolling bands success')
    return d


def get_macd(d, fastperiod=12, slowperiod=26, signalperiod=9):
    d[MACD], d[MACD_SIGNAL], d[MACD_HIST] = talib.MACD(d['close_price'].astype(np.float64),
                                                       fastperiod=fastperiod,
                                                       slowperiod=slowperiod,
                                                       signalperiod=signalperiod)
    logger.info('calculate macd success')
    return d


def get_rsi(d, timeperiod=14):
    d[RSI] = talib.RSI(d['close_price'].values.astype(np.float64), timeperiod=timeperiod)
    logger.info('calculate rsi success')
    return d


def get_kd(d, fastk_period=5, slowk_period=3, slowd_period=3):
    d[K], d[D] = talib.STOCH(d['highest_price'].values.astype(np.float64),
                             d['lowest_price'].astype(np.float64).values,
                             d['close_price'].astype(np.float64).values,
                             fastk_period=fastk_period,
                             slowk_period=slowk_period, slowd_period=slowd_period)
    logger.info('calculate kd success')
    return d


def get_ema(d, timeperiod=30):
    d[EMA] = talib.EMA(d['close_price'].values.astype(np.float64), timeperiod=timeperiod)
    logger.info('calculate ema success')
    return d


def get_wma(d, timeperiod=30):
    d[WMA] = talib.WMA(d['close_price'].values.astype(np.float64), timeperiod=timeperiod)
    logger.info('calculate wma success')
    return d


def get_adx(d, timeperiod=14):
    d[ADX] = talib.ADX(d['highest_price'].values.astype(np.float64),
                       d['lowest_price'].values.astype(np.float64),
                       d['close_price'].values.astype(np.float64), timeperiod=timeperiod)
    logger.info('calculate adx success')
    return d


def get_obv(d):
    d[OBV] = talib.OBV(d['close_price'].values.astype(np.float64),
                       d['volume'].values.astype(np.float64))
    logger.info('calculate obv success')
    return d


def get_candlestick_patterns(d):
    o = d['open_price'].values.astype(np.float64)
    h = d['highest_price'].values.astype(np.float64)
    l = d['lowest_price'].values.astype(np.float64)
    c = d['close_price'].values.astype(np.float64)
    d[CDLHAMMER] = talib.CDLHAMMER(o, h, l, c)  # 锤子线
    d[CDLDOJI] = talib.CDLDOJI(o, h, l, c)  # 十字星
    d[CDLENGULFING] = talib.CDLENGULFING(o, h, l, c)  # 吞没
    d[CDLSHOOTINGSTAR] = talib.CDLSHOOTINGSTAR(o, h, l, c)  # 流星线
    d[CDLINVERTEDHAMMER] = talib.CDLINVERTEDHAMMER(o, h, l, c)  # 倒锤子线
    d[CDLHARAMI] = talib.CDLHARAMI(o, h, l, c)  # 包裹线
    d[CDLMARUBOZU] = talib.CDLMARUBOZU(o, h, l, c)  # 无影线
    d[CDLSPINNINGTOP] = talib.CDLSPINNINGTOP(o, h, l, c)  # 旋转顶部
    logger.info('calculate k_lines pattern success')
    return d


def get_support_resistance_level(d, interval):
    k = 6  # 获取6个支撑/压力线
    day_period = 3
    data_points = interval_to_data_points(interval, day_period)
    recent_prices = d['close_price'][-data_points:].values
    km = KMeans(n_clusters=k)
    km.fit(recent_prices.reshape(-1, 1))
    clusters = km.cluster_centers_

    # 按大小排序，这样可以从低到高得到支撑和压力位
    sorted_clusters = sorted([float(c) for c in clusters])
    return sorted_clusters


def get_ichimoku_clouds(d, tenkan_window=9, kijun_window=26, senkou_window=52, chikou_window=26):
    h = d['highest_price'].astype(np.float64)
    l = d['lowest_price'].astype(np.float64)
    c = d['close_price'].astype(np.float64)

    tenkan_sen = (h.rolling(window=tenkan_window).max() +
                  l.rolling(window=tenkan_window).min()) / 2

    kijun_sen = (h.rolling(window=kijun_window).max() +
                 l.rolling(window=kijun_window).min()) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_window)

    senkou_span_b = ((h.rolling(window=senkou_window).max() +
                      l.rolling(window=senkou_window).min()) / 2).shift(kijun_window)

    chikou_span = c.shift(-chikou_window)

    d[ICHIMOKU_TENKANSEN] = tenkan_sen
    d[ICHIMOKU_KIJUNSEN] = kijun_sen
    d[ICHIMOKU_SENKOU_A] = senkou_span_a
    d[ICHIMOKU_SENKOU_B] = senkou_span_b
    d[ICHIMOKU_CHIKOU] = chikou_span

    logger.info('calculate ichimoku clouds success')
    return d


def calculate_all_technical_indicator(symbol, interval, start_day, end_day):
    try:
        k_lines = get_kline_data(symbol, interval, start_day, end_day)
        d = convert_k_lines_to_df(k_lines)
        tasks = [
            technical_executor.submit(get_bolling_bands, d),
            technical_executor.submit(get_bolling_bands_b, d),
            technical_executor.submit(get_macd, d),
            technical_executor.submit(get_rsi, d),
            technical_executor.submit(get_kd, d),
            technical_executor.submit(get_ema, d),
            technical_executor.submit(get_wma, d),
            technical_executor.submit(get_adx, d),
            technical_executor.submit(get_ichimoku_clouds, d),
            technical_executor.submit(get_obv, d),
            technical_executor.submit(get_candlestick_patterns, d),
        ]
        for task in tasks:
            task.result()

        logger.info('All technical indicators calculated successfully.')
        return d
    except Exception as e:
        logger.error(f"Failed to calculate technical indicators: {str(e)}")
        return None
    finally:
        technical_executor.shutdown()
