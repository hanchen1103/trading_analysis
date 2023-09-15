import json
import os
from datetime import datetime, timedelta
from foundation.Mysql import get_new_cursor
from model.KLineData import KLineData


def get_kline_data(symbol, interval, start=None, end=None):
    filename = f"k_lines_{symbol}_{interval}_{start}_{end}.json"
    dirpath = "../static/cache"
    filepath = os.path.join(dirpath, filename)
    # 计算开始和结束时间的Unix毫秒级时间戳
    kline_data_list = []

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            data_json = json.load(file)
            data = [KLineData.model_validate(d) for d in data_json]
            return data

    current_time = datetime.utcnow()
    if start is not None:
        start_time = current_time - timedelta(days=start)
        start = int(start_time.timestamp() * 1000)
    if end is not None:
        end_time = current_time - timedelta(days=end)
        end = int(end_time.timestamp() * 1000)

    table_name = f"{symbol.lower()}_kline_data_{interval.lower()}"
    query = f"SELECT * FROM {table_name}"
    if start is not None and end is not None:
        query += f" WHERE id BETWEEN {start} AND {end}"

    cursor = get_new_cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()

    for row in result:
        kline_data = KLineData(
            id=row[0],
            open_price=row[1],
            highest_price=row[2],
            lowest_price=row[3],
            close_price=row[4],
            volume=row[5],
            close_time=row[6],
            transaction_amount=row[7],
            transaction_count=row[8],
            active_buy_volume=row[9],
            active_buy_amount=row[10]
        )
        kline_data_list.append(kline_data)

    with open(filepath, 'w', encoding='utf-8') as file:
        serializable_data_list = [item.model_dump() for item in kline_data_list]
        json.dump(serializable_data_list, file, ensure_ascii=False, indent=4)
    return kline_data_list
