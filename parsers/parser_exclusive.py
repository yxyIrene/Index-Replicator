import os
import sys
import re
import pandas as pd
import numpy as np
from typing import List

# 兼容直接单独运行与作为模块导入
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
    清洗股票代码为 6 位字符串：
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
        # 优先提取 6 位连续数字
        m = re.search(r'(\d{6})', val_str)
        if m:
            return m.group(1)
        # 若是纯数字且不足 6 位，向左补齐 0 (如 1 -> 000001)
        digits = re.sub(r'\D', '', val_str)
        if digits and len(digits) <= 6:
            return digits.zfill(6)
        return None

    return series.apply(parse_one)


class ExclusiveMasterParser(BaseParser):
    name = "Exclusive_Master_SBL专属双池解析器"

    def can_handle(self, file_path: str) -> bool:
        """
        独占模式下精准识别 HTI_Inventory：
        1. 文件名包含 hti_inventory 且不含 intraday
        2. 或内部包含 'Available Sources' Tab
        """
        base_name = os.path.basename(file_path).lower()
        if "intraday" in base_name:
            return False
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

        # 1. 优先定位 'Available Sources' Sheet
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
            if any("stock code" in x or "stockcode" in x or "证券代码" in x for x in row_vals):
                header_row = r_idx
                break

        df = pd.read_excel(file_path, sheet_name=target_sheet, header=header_row)
        df.columns = [str(c).strip() for c in df.columns]

        # 3. 列名智能定位
        code_col = next((c for c in df.columns if "stock code" in c.lower() or c == "证券代码"), None)
        name_col = next(
            (c for c in df.columns if "chinese company name" in c.lower() or "公司名称" in c or "证券名称" in c), None)

        # 外部借券列 (Depth)
        ext_qty_col = next((c for c in df.columns if "depth available quantity" in c.lower()), None)
        ext_mv_col = next((c for c in df.columns if "depth available market value" in c.lower()), None)

        # 内部自有券源列 (Available, 且不能带 depth)
        int_qty_col = next((c for c in df.columns if "available quantity" in c.lower() and "depth" not in c.lower()),
                           None)
        int_mv_col = next((c for c in df.columns if "available market value" in c.lower() and "depth" not in c.lower()),
                          None)

        if not code_col:
            raise ValueError(f"文件 {file_path} 中缺少 'Stock Code' 列，现有列: {list(df.columns)}")

        # 4. 数据清洗
        clean_codes = clean_stock_code_series(df[code_col])
        clean_names = df[name_col].astype(str).str.strip().replace("nan", "") if name_col else pd.Series([""] * len(df))

        ext_qtys = clean_numeric_series(df[ext_qty_col]) if ext_qty_col else pd.Series([0.0] * len(df))
        ext_mvs = clean_numeric_series(df[ext_mv_col]) if ext_mv_col else pd.Series([0.0] * len(df))

        int_qtys = clean_numeric_series(df[int_qty_col]) if int_qty_col else pd.Series([0.0] * len(df))
        int_mvs = clean_numeric_series(df[int_mv_col]) if int_mv_col else pd.Series([0.0] * len(df))

        holdings = []

        # 5. 一分为二：生成 external borrow 与 internal borrow 两种持仓
        for idx in range(len(df)):
            code = clean_codes.iloc[idx]
            if not code:
                continue
            name = str(clean_names.iloc[idx])

            # (1) external borrow (深度外部借券)
            ext_q = float(ext_qtys.iloc[idx])
            if ext_q > 0:
                holdings.append(StockHolding(
                    stock_code=code,
                    stock_name=name,
                    quantity=ext_q,
                    market_value=float(ext_mvs.iloc[idx]),
                    fund_name="external borrow"
                ))

            # (2) internal borrow (内部自有券源)
            int_q = float(int_qtys.iloc[idx])
            if int_q > 0:
                holdings.append(StockHolding(
                    stock_code=code,
                    stock_name=name,
                    quantity=int_q,
                    market_value=float(int_mvs.iloc[idx]),
                    fund_name="internal borrow"
                ))

        return holdings


if __name__ == "__main__":
    # 单独运行测试
    test_path = r"\\htisec.local\HK Data\Department\PB\PB\Project Management\D1 Scripts\sbl\output\HTI_Inventory_20260827.xlsx"
    parser = ExclusiveMasterParser()
    print("can_handle:", parser.can_handle(test_path))
    if os.path.exists(test_path):
        res = parser.parse(test_path)
        print(f"✅ 成功提取并拆分出 {len(res)} 条券源记录！")
        df_test = pd.DataFrame([h.__dict__ for h in res])
        print("\n持仓类型统计:")
        print(df_test['fund_name'].value_counts())
        print("\n前 5 条预览:")
        print(df_test[['fund_name', 'stock_code', 'stock_name', 'quantity', 'market_value']].head())
    else:
        print(f"测试文件不存在: {test_path}")