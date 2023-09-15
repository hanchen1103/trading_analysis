# Interval
INTERVAL_1M = '1m'
INTERVAL_3M = '3m'
INTERVAL_5M = '5m'
INTERVAL_15M = '15m'
INTERVAL_30M = '30m'
INTERVAL_1H = '1h'
INTERVAL_2H = '2h'
INTERVAL_4H = '4h'
INTERVAL_6H = '6h'
INTERVAL_8H = '8h'
INTERVAL_12H = '12h'
INTERVAL_1D = '1d'
INTERVAL_3D = '3d'
INTERVAL_1W = '1w'
INTERVAL_1M = '1M'


def interval_to_data_points(interval, day):
    mapping = {
        INTERVAL_1M: 1440,
        INTERVAL_5M: 288,
        INTERVAL_15M: 96,
        INTERVAL_30M: 48,
        INTERVAL_1H: 24,
        INTERVAL_6H: 4,
        INTERVAL_12H: 2,
        INTERVAL_1D: 1,
    }
    return day * mapping.get(interval, 1)