import json
import os

from typing import List

import requests
from pydantic import BaseModel

from foundation.Logger import logger


class BacktestRequest(BaseModel):
    symbol: str
    interval: str
    length: int
    stdDev: float
    volume: int
    startDay: int
    endDay: int


class TradingRecord(BaseModel):
    startTime: str
    endTime: str
    profit: float
    profitPercent: str
    maxProfit: float
    maxProfitTime: str
    maxProfitPercent: str
    isFakeBreak: bool
    tradingStartPrice: float
    tradingEndPrice: float
    tradingAction: str
    investment: float
    startPrice: float
    endPrice: float
    bollingerBandUpper: float
    bollingerBandLower: float


class ResponseData(BaseModel):
    fakeBreakCnt: int
    totalProfit: float
    maxTotalProfit: float
    winRate: float
    maxWinRate: float
    max6WinRate: float
    tradingRecords: List[TradingRecord]
    tradingCnt: int
    trueBreakCnt: int


class BacktestResponse(BaseModel):
    errCode: int
    errMsg: str
    data: ResponseData


def get_bolling_backtest(backtest_request):
    filename = f"bact_test_{backtest_request.symbol}_{backtest_request.interval}_{backtest_request.length}_{backtest_request.stdDev}_{backtest_request.startDay}_{backtest_request.endDay}.json"
    dirpath = "../static/cache"
    filepath = os.path.join(dirpath, filename)

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            data = ResponseData.model_validate(json.load(file))
            return data

    url = "http://43.163.205.166:6669/backtest/bolling"
    # url = "http://127.0.0.1:6669/backtest/bolling"

    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=backtest_request.model_dump(), headers=headers)

    if response.status_code != 200:
        logger.error(f'request err: {response.status_code}, msg: {response.text}')
        return None

    response_obj = BacktestResponse.model_validate_json(response.text)
    if response_obj.errCode != 0:
        logger.error(f'Error in response: errCode: {response_obj.errCode}, errMsg: {response_obj.errMsg}')
        return None

    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(response_obj.data.model_dump(), file, ensure_ascii=False, indent=4)
    return response_obj.data

