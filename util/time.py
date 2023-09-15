from datetime import datetime
from pytz import timezone


def convert_str_to_unix_time_sec(s):
    sgt = timezone('Asia/Shanghai')  # UTC+8
    date_format = "%Y-%m-%d %H:%M:%S"
    date_object = datetime.strptime(s, date_format)
    date_object = sgt.localize(date_object)
    return int(date_object.timestamp())


def convert_str_to_unix_time_mil(s):
    return convert_str_to_unix_time_sec(s) * 1000 + 999
