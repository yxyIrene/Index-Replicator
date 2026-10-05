from dataclasses import dataclass
from typing import Optional

@dataclass
class StockHolding:
    source_file: str = ""             # 源文件名
    fund_name: str = ""               # 提取后的基金名称
    valuation_date: str = ""          # 估值基准日
    stock_code: str = ""              # 股票代码
    stock_name: str = ""              # 股票名称
    quantity: float = 0.0             # 持仓数量
    market_value: Optional[float] = None  # 市值