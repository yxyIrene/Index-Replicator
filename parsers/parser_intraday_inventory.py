import os
import sys
import re
import pandas as pd
import numpy as np
from typing import List

# 兼容独立测试与包导入
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from parsers.base import BaseParser
    from models import StockHolding
else:
    from .base import BaseParser
    from models import StockHolding


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """去除千分位逗号、空格、货币符号后转为浮点数"""
    if series is None:
        return pd.Series([], dtype=float)
    cleaned = (
        series.astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('¥', '', regex=False)
        .str.replace('$', '', regex=False)
        .str.replace(' ', '', regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors='coerce').fillna(0.0)


def clean_stock_code_series(series: pd.Series) -> pd.Series:
    """清洗股票代码为 6 位纯数字"""

    def parse_one(val):
        if pd.isna(val) or str(val).strip() in ["", "nan", "None", "0"]:
            return None
        val_str = str(val).strip()
        if val_str.endswith(".0"):
            val_str = val_str[:-2]
        m = re.search(r'(\d{6})', val_str)
        if m:
            return m.group(1)
        digits = re.sub(r'\D', '', val_str)
        if digits and len(digits) <= 6:
            return digits.zfill(6)
        return None

    return series.apply(parse_one)


class HTIIntradayInventoryParser(BaseParser):
    name = "HTI_Intraday_Inventory盘中券源解析器"

    def __init__(self, exclude_sources: List[str] = None):
        # 完整 7 大来源映射
        self.source_mapping = {
            'INTERNAL': 'Internal',
            'GS': 'OutSource 1',
            'SGCIB': 'OutSource 2',
            'HUATAI': 'OutSource 3',
            'GTHT_PRINCIPAL': 'OutSource 4',
            'GTHT': 'OutSource 5',
            'SWHY': 'OutSource 6'
        }
        self.exclude_sources = [s.upper().strip() for s in (exclude_sources or [])]

    def can_handle(self, file_path: str) -> bool:
        base_name = os.path.basename(file_path).lower()
        if "hti" in base_name and "intraday" in base_name and base_name.endswith(('.xlsx', '.xls')):
            return True
        try:
            excel_file = pd.ExcelFile(file_path)
            for s in excel_file.sheet_names:
                clean_s = re.sub(r'[\s_-]', '', str(s)).lower()
                if "intradaysources" in clean_s or "intraday" in clean_s:
                    return True
        except Exception:
            pass
        return False

    def parse(self, file_path: str, exclude_sources: List[str] = None) -> List[StockHolding]:
        # 合并排除列表
        active_excludes = set(self.exclude_sources)
        if exclude_sources:
            active_excludes.update([s.upper().strip() for s in exclude_sources])

        excel_file = pd.ExcelFile(file_path)

        # 1. 定位 'Intraday Sources' Sheet
        target_sheet = None
        for s in excel_file.sheet_names:
            clean_s = re.sub(r'[\s_-]', '', str(s)).lower()
            if "intradaysources" in clean_s:
                target_sheet = s
                break
        if not target_sheet:
            for s in excel_file.sheet_names:
                if "intraday" in str(s).lower():
                    target_sheet = s
                    break
        if not target_sheet:
            target_sheet = excel_file.sheet_names[0]

        # 2. 定位表头所在行
        df_raw = pd.read_excel(file_path, sheet_name=target_sheet, header=None, nrows=5)
        header_row = 0
        for r_idx in range(len(df_raw)):
            row_vals = [str(x).strip().lower() for x in df_raw.iloc[r_idx].values]
            if any("stock code" in x or "closing price" in x for x in row_vals):
                header_row = r_idx
                break

        df = pd.read_excel(file_path, sheet_name=target_sheet, header=header_row)
        df.columns = [str(c).strip() for c in df.columns]

        # 3. 基础列定位
        code_col = next((c for c in df.columns if "stock code" in c.lower() or c == "证券代码"), None)
        price_col = next((c for c in df.columns if "closing price" in c.lower() or "收盘价" in c), None)
        name_col = next(
            (c for c in df.columns if "chinese company name" in c.lower() or "公司名称" in c or "证券名称" in c), None)

        if not code_col or not price_col:
            raise ValueError(f"Intraday 文件 {file_path} 缺少 'Stock Code' 或 'Closing Price' 列")

        clean_codes = clean_stock_code_series(df[code_col])
        clean_prices = clean_numeric_series(df[price_col])
        clean_names = df[name_col].astype(str).str.strip().replace("nan", "") if name_col else pd.Series([""] * len(df))

        holdings = []

        # 4. 遍历券源市值列（自动跳过被排除的分类）
        for fund_label, col_keyword in self.source_mapping.items():
            # 🎯 核心排除判断
            if fund_label.upper() in active_excludes:
                continue

            target_mv_col = next(
                (c for c in df.columns if col_keyword.lower() in c.lower() and "market value" in c.lower()), None)
            if not target_mv_col:
                target_mv_col = next(
                    (c for c in df.columns if col_keyword.lower() in c.lower() and "value" in c.lower()), None)

            if not target_mv_col:
                continue

            clean_mvs = clean_numeric_series(df[target_mv_col])

            for idx in range(len(df)):
                code = clean_codes.iloc[idx]
                if not code:
                    continue

                mv = float(clean_mvs.iloc[idx])
                price = float(clean_prices.iloc[idx])

                if mv > 0 and price > 0:
                    qty = mv / price
                    holdings.append(StockHolding(
                        stock_code=code,
                        stock_name=str(clean_names.iloc[idx]),
                        quantity=float(qty),
                        market_value=mv,
                        fund_name=fund_label
                    ))

        return holdings