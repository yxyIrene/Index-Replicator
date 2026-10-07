# replication/data_loaders.py
import os
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from parsers.parser_intraday_inventory import HTIIntradayInventoryParser

def fetch_stock_market_cap_from_db(conn_str: str, stock_codes: list, trade_date: str) -> pd.DataFrame:
    """提取个股历史/最新市值"""
    engine = create_engine(conn_str)
    formatted_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
    codes_str = "','".join(stock_codes)
    query = text(f"""
        WITH ranked_cap AS (
            SELECT code AS stock_code, market_cap, trade_date,
                   ROW_NUMBER() OVER(PARTITION BY code ORDER BY trade_date DESC) as rn
            FROM public.md_stock_rk_info
            WHERE code IN ('{codes_str}')
              AND trade_date <= TO_DATE('{formatted_date}', 'YYYY-MM-DD')
              AND market_cap IS NOT NULL
        )
        SELECT stock_code, market_cap FROM ranked_cap WHERE rn = 1;
    """)
    with engine.connect() as conn:
        df_cap = pd.read_sql(query, conn)
    df_cap['stock_code'] = df_cap['stock_code'].astype(str).str.zfill(6)
    df_cap['market_cap'] = pd.to_numeric(df_cap['market_cap'], errors='coerce')
    return df_cap


def fetch_st_stocks_from_db(conn_str: str, trade_date: str) -> set:
    """提取 ST/*ST 标的"""
    engine = create_engine(conn_str)
    formatted_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
    query = text(f"""
        WITH ranked_st AS (
            SELECT code, name, effective_date,
                   ROW_NUMBER() OVER(PARTITION BY code ORDER BY effective_date DESC) as rn
            FROM public.md_stock_st
            WHERE effective_date <= TO_DATE('{formatted_date}', 'YYYY-MM-DD')
        )
        SELECT code, name FROM ranked_st 
        WHERE rn = 1 AND (name LIKE '%ST%' OR name LIKE '%*ST%');
    """)
    try:
        with engine.connect() as conn:
            df_st = pd.read_sql(query, conn)
        if df_st.empty:
            return set()
        return set(df_st['code'].astype(str).str.extract(r'(\d{6})')[0].dropna().str.zfill(6))
    except Exception as e:
        print(f"⚠️ 从 md_stock_st 提取 ST 标的失败: {e}")
        return set()


def load_jump_restriction_list(share_dir: str, trade_date: str) -> set:
    """从 Jump 共享盘读取限制名单"""
    if not os.path.exists(share_dir):
        print(f"⚠️ 无法访问共享盘路径: {share_dir}")
        return set()
    target_fp = None
    for fname in os.listdir(share_dir):
        if not fname.startswith(('~', '.')) and trade_date in fname and "restriction" in fname.lower():
            target_fp = os.path.join(share_dir, fname)
            break
    if not target_fp:
        return set()
    try:
        df_res = pd.read_csv(target_fp, dtype=str) if target_fp.lower().endswith('.csv') else pd.read_excel(target_fp, dtype=str)
        df_res.columns = [str(c).strip().upper() for c in df_res.columns]
        code_col = next((c for c in df_res.columns if 'TICKER' in c or 'RIC' in c), None)
        if not code_col:
            return set()
        codes = df_res[code_col].dropna().astype(str).str.extract(r'(\d{6})')[0].dropna().tolist()
        return set(c.zfill(6) for c in codes)
    except Exception as e:
        print(f"❌ 读取限制名单失败: {e}")
        return set()


def load_bics_industry_mapping(trade_date: str, ind_dir: str = "Industry Mappings") -> pd.DataFrame:
    """加载 BICS 行业映射表"""
    target_fp = os.path.join(ind_dir, f"industry_mapping_{trade_date}.xlsx")
    if not os.path.exists(target_fp):
        candidates = [os.path.join(ind_dir, f) for f in os.listdir(ind_dir) if f.startswith("industry_mapping") and f.endswith(".xlsx")]
        if not candidates:
            raise FileNotFoundError(f"❌ 未找到 industry_mapping 文件！")
        target_fp = sorted(candidates)[-1]
    df_ind = pd.read_excel(target_fp, dtype={'stock_code': str})
    df_ind['stock_code'] = df_ind['stock_code'].astype(str).str.zfill(6)
    df_ind['bics_level_2'] = df_ind['bics_level_2'].fillna('Others').astype(str).str.strip()
    return df_ind[['stock_code', 'bics_level_2']]


def load_located_list(trade_date: str, candidate_dirs: list) -> pd.DataFrame:
    """读取并聚合锁券清单"""
    located_file = None
    target_names = ["Located List.xlsx", "Located List.xls", f"Located_List_{trade_date}.xlsx"]
    for sp in ["."] + candidate_dirs:
        if not os.path.exists(sp): continue
        for name in target_names:
            fp = os.path.join(sp, name)
            if os.path.isfile(fp):
                located_file = os.path.abspath(fp)
                break
        if located_file: break

    if not located_file:
        return pd.DataFrame(columns=['stock_code', 'Located Qty', 'WA borrow cost'])

    df_loc = pd.read_excel(located_file, dtype=str)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]
    code_col = next((c for c in df_loc.columns if '证券代码' in c), None)
    qty_col = next((c for c in df_loc.columns if '合约数量' in c), None)
    rate_col = next((c for c in df_loc.columns if any(k in c.lower() for k in ['利率', '费率', 'rate', 'cost'])), None)

    if not code_col or not qty_col:
        return pd.DataFrame(columns=['stock_code', 'Located Qty', 'WA borrow cost'])

    df_loc['stock_code'] = df_loc[code_col].astype(str).str.strip().str[:6].str.zfill(6)
    df_loc['qty'] = pd.to_numeric(df_loc[qty_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
    df_loc['rate'] = pd.to_numeric(df_loc[rate_col].astype(str).str.replace('%', '').str.replace(',', '').str.strip(), errors='coerce').fillna(0.0) if rate_col else 0.0
    df_loc['qty_x_rate'] = df_loc['qty'] * df_loc['rate']

    grouped = df_loc.dropna(subset=['stock_code']).groupby('stock_code', as_index=False).agg(
        total_qty=('qty', 'sum'),
        sum_qty_rate=('qty_x_rate', 'sum')
    )
    grouped['WA borrow cost'] = np.where(grouped['total_qty'] > 0, grouped['sum_qty_rate'] / grouped['total_qty'], 0.0)
    return grouped[['stock_code', 'total_qty', 'WA borrow cost']].rename(columns={'total_qty': 'Located Qty'})


def load_internal_inventory(source_dirs: list, trade_date: str, exclude_sources: list) -> pd.DataFrame:
    """提取内部底仓可用量"""
    parser = HTIIntradayInventoryParser(exclude_sources=exclude_sources)
    all_holdings = []
    for val_dir in source_dirs:
        if not os.path.exists(val_dir): continue
        for fn in os.listdir(val_dir):
            if "intraday" in fn.lower() and trade_date in fn and fn.endswith(('.xlsx', '.xls')):
                try:
                    all_holdings.extend(parser.parse(os.path.join(val_dir, fn), exclude_sources=exclude_sources))
                except Exception: pass
    if not all_holdings:
        return pd.DataFrame(columns=['stock_code', 'internal_avail_qty'])
    df_holdings = pd.DataFrame([h.__dict__ for h in all_holdings])
    df_holdings['stock_code'] = df_holdings['stock_code'].astype(str).str.zfill(6)
    df_holdings['quantity'] = pd.to_numeric(df_holdings['quantity'], errors='coerce').fillna(0.0)
    return df_holdings.groupby('stock_code', as_index=False)['quantity'].sum().rename(columns={'quantity': 'internal_avail_qty'})