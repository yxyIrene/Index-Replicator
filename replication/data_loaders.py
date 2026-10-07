# replication/data_loaders.py
import os
import re
import warnings
from typing import Dict, List, Optional, Set

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.engine import Engine

from parsers.parser_intraday_inventory import HTIIntradayInventoryParser


# =========================================================================
# 0. 引擎缓存（模块级单例，避免每次 create_engine 重复建连接池）
# =========================================================================
_ENGINE_CACHE: Dict[str, Engine] = {}


def get_engine(conn_str: str) -> Engine:
    """复用 SQLAlchemy Engine。全项目统一从该函数取 Engine。"""
    if conn_str not in _ENGINE_CACHE:
        _ENGINE_CACHE[conn_str] = create_engine(
            conn_str, pool_pre_ping=True, pool_size=10, max_overflow=20
        )
    return _ENGINE_CACHE[conn_str]


# =========================================================================
# 通用工具
# =========================================================================
def _fmt_date(trade_date: str) -> str:
    """YYYYMMDD -> YYYY-MM-DD"""
    return f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"


def _to_6digit_codes(series: pd.Series) -> Set[str]:
    """从任意格式的代码列中抽取 6 位数字代码集合"""
    return set(
        series.astype(str).str.extract(r'(\d{6})')[0]
        .dropna().str.zfill(6).tolist()
    )


# =========================================================================
# [新增] 停牌数据（A 股 / 港股统一表 md_stock_suspension）
# =========================================================================
def fetch_suspension_stocks_from_db(
    conn_str: str,
    trade_date: str,
    exchange: Optional[str] = None,
) -> Set[str]:
    """
    从 public.md_stock_suspension 获取指定交易日的停牌股票代码集合。

    参数:
        conn_str    : 数据库连接串
        trade_date  : YYYYMMDD
        exchange    : 可选，'SSE' / 'SZSE' / 'HKEX' 等；None 表示不过滤交易所

    兜底策略（重要）:
        该表可能只有近期数据，历史回测查不到数据 / 表不存在 / 连接异常时，
        一律返回空集合并打印警告；下游依赖"行情价格缺失"作为兜底。
    """
    engine = get_engine(conn_str)
    fmt_date = _fmt_date(trade_date)

    sql = """
        SELECT code, exchange, trade_status
        FROM public.md_stock_suspension
        WHERE trade_date = :trade_date
    """
    params: Dict[str, str] = {"trade_date": fmt_date}
    if exchange:
        sql += " AND exchange = :exchange"
        params["exchange"] = exchange

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
    except Exception as e:
        warnings.warn(
            f"⚠️ 获取 {trade_date} 停牌数据失败（历史数据缺失 / 表不存在 / 连接异常）: {e}；"
            f"本次降级为空集，下游以行情价格缺失兜底。"
        )
        return set()

    if df.empty:
        return set()

    # 仅保留明确标注"停牌"的记录
    if 'trade_status' in df.columns:
        df = df[df['trade_status'].astype(str).str.contains('停牌', na=False)]

    return _to_6digit_codes(df['code'])


# =========================================================================
# 市值（参数化）
# =========================================================================
def fetch_stock_market_cap_from_db(
    conn_str: str, stock_codes: List[str], trade_date: str
) -> pd.DataFrame:
    """提取个股历史/最新市值（SQL 参数化，避免注入）"""
    engine = get_engine(conn_str)
    fmt_date = _fmt_date(trade_date)

    # 归一化代码
    codes = [str(c).strip().zfill(6)[-6:] for c in stock_codes if str(c).strip()]
    if not codes:
        return pd.DataFrame(columns=['stock_code', 'market_cap'])

    query = text("""
        WITH ranked_cap AS (
            SELECT code AS stock_code, market_cap, trade_date,
                   ROW_NUMBER() OVER(PARTITION BY code ORDER BY trade_date DESC) as rn
            FROM public.md_stock_rk_info
            WHERE code IN :codes
              AND trade_date <= :fmt_date
              AND market_cap IS NOT NULL
        )
        SELECT stock_code, market_cap FROM ranked_cap WHERE rn = 1;
    """).bindparams(bindparam('codes', expanding=True))

    with engine.connect() as conn:
        df_cap = pd.read_sql(
            query, conn, params={"codes": codes, "fmt_date": fmt_date}
        )

    if df_cap.empty:
        return pd.DataFrame(columns=['stock_code', 'market_cap'])

    df_cap['stock_code'] = df_cap['stock_code'].astype(str).str.zfill(6)
    df_cap['market_cap'] = pd.to_numeric(df_cap['market_cap'], errors='coerce')
    return df_cap


# =========================================================================
# ST 标的（参数化）
# =========================================================================
def fetch_st_stocks_from_db(conn_str: str, trade_date: str) -> Set[str]:
    """提取 ST/*ST 标的"""
    engine = get_engine(conn_str)
    fmt_date = _fmt_date(trade_date)

    query = text("""
        WITH ranked_st AS (
            SELECT code, name, effective_date,
                   ROW_NUMBER() OVER(PARTITION BY code ORDER BY effective_date DESC) as rn
            FROM public.md_stock_st
            WHERE effective_date <= :fmt_date
        )
        SELECT code, name FROM ranked_st 
        WHERE rn = 1 AND (name LIKE '%ST%' OR name LIKE '%*ST%');
    """)
    try:
        with engine.connect() as conn:
            df_st = pd.read_sql(query, conn, params={"fmt_date": fmt_date})
        if df_st.empty:
            return set()
        return _to_6digit_codes(df_st['code'])
    except Exception as e:
        print(f"⚠️ 从 md_stock_st 提取 ST 标的失败: {e}")
        return set()


# =========================================================================
# 以下为文件 I/O，行为不变
# =========================================================================
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