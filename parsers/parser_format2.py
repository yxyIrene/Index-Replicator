import re
import pandas as pd
from typing import List
from models import StockHolding
from parsers.base import BaseParser


class Format2Parser(BaseParser):
    """
    适用于：神州稳健对账单
    规则：表头在第20行，第一列为证券代码，取'可用数量'列，向下遇到合计行结束
    """
    name = "格式二(神州对账单)"

    def can_handle(self, file_path: str) -> bool:
        if not file_path.lower().endswith(('.xlsx', '.xls', '.csv')):
            return False
        try:
            if file_path.lower().endswith('.csv'):
                df_preview = pd.read_csv(file_path, nrows=25, header=None)
            else:
                df_preview = pd.read_excel(file_path, nrows=25, header=None)

            content = df_preview.to_string()
            return "证券代码" in content and "可用数量" in content
        except Exception:
            return False

    def parse(self, file_path: str) -> List[StockHolding]:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path, header=19, dtype={'证券代码': str})
        else:
            df = pd.read_excel(file_path, header=19, dtype={'证券代码': str})

        df.columns = [str(col).strip() for col in df.columns]

        if "证券代码" not in df.columns or "可用数量" not in df.columns:
            raise ValueError(f"文件 {file_path} 缺少 '证券代码' 或 '可用数量' 列")

        holdings: List[StockHolding] = []

        for _, row in df.iterrows():
            code_raw = str(row.get("证券代码", "")).strip()
            name_raw = str(row.get("证券简称", "")).strip()
            qty_raw = row.get("可用数量", 0)

            if "合计" in code_raw or "合计" in name_raw or "总计" in code_raw:
                break
            if not code_raw or code_raw == "nan":
                continue

            clean_code = code_raw.split('.')[0]
            if clean_code.isdigit() and len(clean_code) <= 6:
                clean_code = clean_code.zfill(6)

            if re.match(r"^\d{6}$", clean_code):
                try:
                    quantity = float(str(qty_raw).replace(",", "").strip() or 0)
                except (ValueError, TypeError):
                    quantity = 0.0

                market_val = row.get("当前市值", None)
                try:
                    market_val = float(str(market_val).replace(",", "").strip()) if pd.notna(market_val) else None
                except (ValueError, TypeError):
                    market_val = None

                # 统一使用显式关键字传参
                holdings.append(StockHolding(
                    stock_code=clean_code,
                    stock_name=name_raw,
                    quantity=quantity,
                    market_value=market_val
                ))

        return holdings