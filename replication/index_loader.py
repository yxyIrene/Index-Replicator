import os
import re
from typing import Dict, Optional
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# =========================================================================
# 指数代码映射关系 (业务 Sheet 简称 -> TAICHI 库对应代码)
# =========================================================================
INDEX_DB_MAPPING = {
    "SHSN300": {"index_code": "000300.SH", "daily_code": "000300"},  # 沪深300
    "CSI1000": {"index_code": "000852.SH", "daily_code": "000852"},  # 中证1000
    "SSE50": {"index_code": "000016.SH", "daily_code": "000016"},  # 上证50
    "SH000905": {"index_code": "000905.SH", "daily_code": "000905"},  # 中证500
    "SH932000": {"index_code": "932000.CSI", "daily_code": "932000"}  # 中证2000
}

TARGET_SHEETS = ["SHSN300", "CSI1000", "SSE50", "SH000905", "SH932000"]


# =========================================================================
# 1. Excel 文件解析函数 (精确解析共享盘权威文件)
# =========================================================================
def load_all_indices_from_file(file_path: str) -> Dict[str, pd.DataFrame]:
    """从本地或共享盘 Excel 文件中读取各指数成分与真实权重"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到指数文件: {file_path}")

    excel_file = pd.ExcelFile(file_path)
    available_sheets = excel_file.sheet_names

    indices_data = {}

    for sheet_name in TARGET_SHEETS:
        matched_sheet = next((s for s in available_sheets if s.strip().upper() == sheet_name.upper()), None)
        if not matched_sheet:
            continue

        df = pd.read_excel(excel_file, sheet_name=matched_sheet, header=1)
        df.columns = [str(c).strip() for c in df.columns]

        # 提取 6 位纯数字代码
        df['stock_code'] = df['Ticker'].astype(str).str.extract(r'(\d{6})')[0].str.zfill(6)

        # 权重归一化处理
        df['weight'] = pd.to_numeric(df['Weights'], errors='coerce').fillna(0.0)
        if df['weight'].sum() > 1.5:
            df['weight'] = df['weight'] / 100.0
        total_w = df['weight'].sum()
        if total_w > 0:
            df['weight'] = df['weight'] / total_w

        df['price'] = pd.to_numeric(df['Value Price'], errors='coerce').fillna(1.0)
        df['stock_name'] = df['Name'].astype(str).str.strip() if 'Name' in df.columns else ""
        df['lot_size'] = pd.to_numeric(df['Lot size/Min order size'], errors='coerce').fillna(100).astype(int)

        df_clean = df.dropna(subset=['stock_code'])[['stock_code', 'stock_name', 'weight', 'price', 'lot_size']].copy()
        indices_data[sheet_name] = df_clean

    return indices_data


# 保持向下兼容旧命名
load_all_indices = load_all_indices_from_file


# =========================================================================
# 2. 数据库兜底解析函数 (月末快照 + Pt/P0 动态漂移修正)
# =========================================================================
def load_all_indices_from_db(conn_str: str, trade_date: str) -> Dict[str, pd.DataFrame]:
    """从 TAICHI 数据库提取成分，并结合当日最新收盘价进行权重自然漂移重算"""
    clean_date = re.sub(r'[^0-9]', '', str(trade_date))
    formatted_date = f"{clean_date[:4]}-{clean_date[4:6]}-{clean_date[6:8]}"

    target_codes = [v['index_code'] for v in INDEX_DB_MAPPING.values()]
    codes_str = "','".join(target_codes)

    query = text(f"""
        WITH latest_snap AS (
            SELECT index_code, MAX(trade_date) as snap_date
            FROM public.md_index_weight
            WHERE trade_date <= TO_DATE('{formatted_date}', 'YYYY-MM-DD')
              AND index_code IN ('{codes_str}')
            GROUP BY index_code
        ),
        ranked_weights AS (
            SELECT 
                w.index_code,
                w.code AS stock_code,
                w.name AS stock_name,
                w.i_weight AS base_weight,
                w.trade_date AS snap_date,
                COALESCE(d0.close_price, d0.pre_close, 1.0) AS price_t0,
                COALESCE(dt.close_price, dt.pre_close, d0.close_price, 1.0) AS price_tt,
                ROW_NUMBER() OVER (
                    PARTITION BY w.index_code, w.code 
                    ORDER BY w.trade_date DESC, w.update_date DESC
                ) as rn
            FROM public.md_index_weight w
            JOIN latest_snap ls 
                ON w.index_code = ls.index_code 
                AND w.trade_date = ls.snap_date
            LEFT JOIN public.md_stock_daily d0 
                ON w.code = d0.code 
                AND d0.trade_date = ls.snap_date
            LEFT JOIN public.md_stock_daily dt 
                ON w.code = dt.code 
                AND dt.trade_date = TO_DATE('{formatted_date}', 'YYYY-MM-DD')
        )
        SELECT 
            index_code,
            stock_code,
            stock_name,
            base_weight,
            price_t0,
            price_tt AS price,
            100 AS lot_size
        FROM ranked_weights
        WHERE rn = 1;
    """)

    engine = create_engine(conn_str)
    with engine.connect() as conn:
        df_all = pd.read_sql(query, conn)

    if df_all.empty:
        raise ValueError(f"❌ TAICHI 数据库在 {trade_date} ({formatted_date}) 之前未找到任何指数成分快照！")

    indices_dict = {}

    for sheet_name, cfg in INDEX_DB_MAPPING.items():
        db_code = cfg['index_code']
        df_sub = df_all[df_all['index_code'] == db_code].copy()
        if df_sub.empty:
            continue

        df_sub['stock_code'] = df_sub['stock_code'].astype(str).str.extract(r'(\d{6})')[0].str.zfill(6)

        df_sub['price_t0'] = pd.to_numeric(df_sub['price_t0'], errors='coerce').fillna(1.0).apply(
            lambda x: x if x > 0.01 else 1.0)
        df_sub['price'] = pd.to_numeric(df_sub['price'], errors='coerce').fillna(df_sub['price_t0']).apply(
            lambda x: x if x > 0.01 else 1.0)

        # 1. 转换数值，非数值或空值自动转为 NaN
        df_sub['base_weight'] = pd.to_numeric(df_sub['base_weight'], errors='coerce')

        # 2. 🎯 核心清洗：剔除权重为 NaN 或 <= 0 的脏数据行（精准清除那 6 支空壳股）
        df_sub = df_sub.dropna(subset=['base_weight']).copy()
        df_sub = df_sub[df_sub['base_weight'] > 0].copy()

        # 3. 正常做权重百分比换算与去重
        if df_sub['base_weight'].sum() > 1.5:
            df_sub['base_weight'] = df_sub['base_weight'] / 100.0

        df_sub = df_sub.sort_values(by='base_weight', ascending=False).drop_duplicates(subset=['stock_code'],
                                                                                       keep='first')

        # 4. 价格漂移修正与归一化
        drift_factor = df_sub['price'] / df_sub['price_t0']
        df_sub['weight'] = df_sub['base_weight'] * drift_factor

        tot_w = df_sub['weight'].sum()
        if tot_w > 0:
            df_sub['weight'] = df_sub['weight'] / tot_w

        df_sub['stock_name'] = df_sub['stock_name'].fillna("")
        df_sub['lot_size'] = 100

        df_clean = df_sub.dropna(subset=['stock_code'])[
            ['stock_code', 'stock_name', 'weight', 'price', 'lot_size']].copy()
        indices_dict[sheet_name] = df_clean

    return indices_dict


# =========================================================================
# 3. 方案 2 核心入口：优先读取共享盘文件，缺失则自动切 DB 漂移兜底
# =========================================================================
def load_all_indices_smart(
        trade_date: str,
        conn_str: str,
        index_share_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    智能双轨加载器:
    1. 优先在共享盘中寻找 `Index Data {trade_date}.xlsx`
    2. 若文件不存在或共享盘断开，自动平滑切换至 TAICHI 数据库 (月末快照 + 价格漂移)
    """
    clean_date = re.sub(r'[^0-9]', '', str(trade_date))

    # 1. 尝试匹配共享盘文件
    if index_share_dir and os.path.exists(index_share_dir):
        # 兼容几种常见文件名变体与扩展名
        file_candidates = [
            os.path.join(index_share_dir, f"Index Data {clean_date}.xlsx"),
            os.path.join(index_share_dir, f"Index Data {clean_date}.xls"),
            os.path.join(index_share_dir, f"Index_{clean_date}.xlsx"),
        ]

        for file_path in file_candidates:
            if os.path.exists(file_path):
                print(f"📁 [{clean_date}] BBG Index file found: {os.path.basename(file_path)}")
                try:
                    return load_all_indices_from_file(file_path)
                except Exception as e:
                    print(f"⚠️ 解析共享盘文件失败 ({e})，正在自动切换数据库兜底...")
                    break

    # 2. 兜底回退到数据库
    print(f"🌐 [{clean_date}] 无当日指数文件，使用 TAICHI 数据库 (月末快照 + 动态价格漂移) 自动补全...")
    return load_all_indices_from_db(conn_str, clean_date)


# =========================================================================
# 4. 指数点位提取函数 (优先 DB，确保准确)
# =========================================================================
def get_index_spot_price_from_db(conn_str: str, matched_name: str, trade_date: str) -> float:
    """提取指数收盘点位"""
    clean_date = re.sub(r'[^0-9]', '', str(trade_date))
    formatted_date = f"{clean_date[:4]}-{clean_date[4:6]}-{clean_date[6:8]}"

    cfg = INDEX_DB_MAPPING.get(matched_name.upper())
    if not cfg:
        key = next((k for k in INDEX_DB_MAPPING if k in matched_name.upper()), None)
        cfg = INDEX_DB_MAPPING[key] if key else None

    if not cfg:
        return 1.0

    daily_code = cfg['daily_code']

    query = text(f"""
        SELECT close_price 
        FROM public.md_index_daily 
        WHERE code = '{daily_code}' 
          AND trade_date <= TO_DATE('{formatted_date}', 'YYYY-MM-DD')
          AND close_price IS NOT NULL
        ORDER BY trade_date DESC
        LIMIT 1;
    """)

    engine = create_engine(conn_str)
    with engine.connect() as conn:
        df_p = pd.read_sql(query, conn)

    if not df_p.empty and pd.notna(df_p['close_price'].iloc[0]):
        return float(df_p['close_price'].iloc[0])

    return 1.0