import os
import sys
import re
import pandas as pd
import numpy as np
from typing import List

# 兼容独立运行与包导入
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
    """
    清洗股票代码：
    - 兼容 600519.0 -> 600519
    - 兼容 整数 1 -> 000001
    - 兼容 '000001 CH', '600519.SH'
    """

    def parse_one(val):
        if pd.isna(val) or str(val).strip() in ["", "nan", "None", "0"]:
            return None
        val_str = str(val).strip()
        if val_str.endswith(".0"):
            val_str = val_str[:-2]
        # 提取6位连续数字
        m = re.search(r'(\d{6})', val_str)
        if m:
            return m.group(1)
        # 若是纯数字且小于6位，左补0 (如平安 1 -> 000001)
        digits = re.sub(r'\D', '', val_str)
        if digits and len(digits) <= 6:
            return digits.zfill(6)
        return None

    return series.apply(parse_one)


class HTIInventoryParser(BaseParser):
    name = "HTI_Inventory_SBL券源解析器"

    def can_handle(self, file_path: str) -> bool:
        base_name = os.path.basename(file_path).lower()
        if "hti_inventory" in base_name or "hti inventory" in base_name:
            return True
        try:
            excel_file = pd.ExcelFile(file_path)
            for s in excel_file.sheet_names:
                clean_s = re.sub(r'[\s_-]', '', str(s)).lower()
                if "availablesources" in clean_s or "available" in clean_s:
                    return True
        except Exception:
            pass
        return False

    def parse(self, file_path: str) -> List[StockHolding]:
        excel_file = pd.ExcelFile(file_path)

        # 1. 匹配 Available Sources Sheet
        target_sheet = None
        for s in excel_file.sheet_names:
            clean_s = re.sub(r'[\s_-]', '', str(s)).lower()
            if "availablesources" in clean_s:
                target_sheet = s
                break
        if not target_sheet:
            for s in excel_file.sheet_names:
                if "available" in str(s).lower():
                    target_sheet = s
                    break
        if not target_sheet:
            target_sheet = excel_file.sheet_names[0]

        # 2. 自动定位表头行 (前 5 行内查找包含 'Stock Code' 的行)
        df_raw = pd.read_excel(file_path, sheet_name=target_sheet, header=None, nrows=5)
        header_row = 0
        for r_idx in range(len(df_raw)):
            row_vals = [str(x).strip().lower() for x in df_raw.iloc[r_idx].values]
            if any("stock code" in x or "stockcode" in x for x in row_vals):
                header_row = r_idx
                break

        df = pd.read_excel(file_path, sheet_name=target_sheet, header=header_row)
        df.columns = [str(c).strip() for c in df.columns]

        # 3. 列名智能定位
        code_col = next(
            (c for c in df.columns if "stock code" in c.lower() or "stockcode" in c.lower() or c == "证券代码"), None)
        qty_col = next((c for c in df.columns if "depth available quantity" in c.lower()), None)
        # 如果没有 Depth 列，回退取 Available Quantity
        if not qty_col:
            qty_col = next((c for c in df.columns if "available quantity" in c.lower() or "可借数量" in c), None)

        mv_col = next((c for c in df.columns if "depth available market value" in c.lower()), None)
        if not mv_col:
            mv_col = next((c for c in df.columns if "available market value" in c.lower() or "可借市值" in c), None)

        name_col = next(
            (c for c in df.columns if "chinese company name" in c.lower() or "公司名称" in c or "证券名称" in c), None)

        if not code_col or not qty_col:
            raise ValueError(f"无法在文件 {file_path} 中定位代码列或数量列，现有列: {list(df.columns)}")

        # 4. 清洗提取
        clean_codes = clean_stock_code_series(df[code_col])
        clean_qtys = clean_numeric_series(df[qty_col])
        clean_mvs = clean_numeric_series(df[mv_col]) if mv_col else pd.Series([0.0] * len(df))

        if name_col:
            clean_names = df[name_col].astype(str).str.strip().replace("nan", "")
        else:
            clean_names = pd.Series([""] * len(df))

        # 5. 过滤有效数据 (代码有效 且 数量 > 0)
        valid_mask = clean_codes.notna() & (clean_qtys > 0)
        df_valid = pd.DataFrame({
            'code': clean_codes[valid_mask],
            'name': clean_names[valid_mask],
            'qty': clean_qtys[valid_mask],
            'mv': clean_mvs[valid_mask]
        })

        holdings = []
        for _, row in df_valid.iterrows():
            holdings.append(StockHolding(
                stock_code=str(row['code']).zfill(6),
                stock_name=str(row['name']),
                quantity=float(row['qty']),
                market_value=float(row['mv'])
            ))

        return holdings


if __name__ == "__main__":
    # 单独运行测试脚本
    test_path = r"\\htisec.local\HK Data\Department\PB\PB\Project Management\D1 Scripts\sbl\output\HTI_Inventory_20260826.xlsx"
    parser = HTIInventoryParser()
    print("can_handle:", parser.can_handle(test_path))
    if os.path.exists(test_path):
        res = parser.parse(test_path)
        print(f"✅ 成功解析 {len(res)} 条券源持仓！前 5 条预览:")
        for item in res[:5]:
            print(item.__dict__)
    else:
        print(f"测试路径未找到: {test_path}")