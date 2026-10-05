import os
import pandas as pd
from typing import Dict

# 目标 4 个 Sheet
TARGET_SHEETS = ["SHSN300", "CSI1000", "SSE50", "SH000905","SH932000"]


def load_all_indices(file_path: str) -> Dict[str, pd.DataFrame]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到指数文件: {file_path}")

    excel_file = pd.ExcelFile(file_path)
    available_sheets = excel_file.sheet_names

    indices_data = {}

    for sheet_name in TARGET_SHEETS:
        # 匹配 sheet 名（忽略空格与大小写）
        matched_sheet = next((s for s in available_sheets if s.strip().upper() == sheet_name.upper()), None)
        if not matched_sheet:
            continue

        # 表头在第二行 -> header=1
        df = pd.read_excel(excel_file, sheet_name=matched_sheet, header=1)
        # 去除列名两端空格
        df.columns = [str(c).strip() for c in df.columns]

        # 1. 提取 6 位数字代码 (如 '000009 CH' -> '000009')
        df['stock_code'] = df['Ticker'].astype(str).str.extract(r'(\d{6})')[0].str.zfill(6)

        # 2. 权重处理（Bloomberg 格式权重归一化）
        df['weight'] = pd.to_numeric(df['Weights'], errors='coerce').fillna(0.0)
        if df['weight'].sum() > 1.5:
            df['weight'] = df['weight'] / 100.0
        total_w = df['weight'].sum()
        if total_w > 0:
            df['weight'] = df['weight'] / total_w

        # 3. 市价、名称、每手股数
        df['price'] = pd.to_numeric(df['Value Price'], errors='coerce').fillna(1.0)
        df['stock_name'] = df['Name'].astype(str).str.strip() if 'Name' in df.columns else ""
        df['lot_size'] = pd.to_numeric(df['Lot size/Min order size'], errors='coerce').fillna(100).astype(int)

        # 过滤空值并返回干净的列
        df_clean = df.dropna(subset=['stock_code'])[['stock_code', 'stock_name', 'weight', 'price', 'lot_size']].copy()

        indices_data[sheet_name] = df_clean

    return indices_data