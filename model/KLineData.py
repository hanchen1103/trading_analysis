from pydantic import BaseModel


class KLineData(BaseModel):
    id: int
    open_price: float
    highest_price: float
    lowest_price: float
    close_price: float
    volume: float
    close_time: int
    transaction_amount: float
    transaction_count: int
    active_buy_volume: float
    active_buy_amount: float
