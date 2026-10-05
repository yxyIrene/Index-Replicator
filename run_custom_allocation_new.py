import os
import sys
import re
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import linprog
from sqlalchemy import create_engine, text

from parsers.parser_intraday_inventory import HTIIntradayInventoryParser
from replication.locate_demand_generator import generate_dual_locate_demand_lists
from replication.date_utils import get_previous_trading_day, PathConfig
from replication.index_loader import load_all_indices, TARGET_SHEETS

# ==============================================================================
# 🎯 1. 核心系统、风控与业务参数配置
# ==============================================================================
SOURCE_PRIORITY = ['INTERNAL', 'GTHT_PRINCIPAL', 'GTHT', 'GS', 'SGCIB', 'HUATAI', 'SWHY']
EXCLUDE_SOURCES = ['GTHT', 'GS', 'SGCIB', 'HUATAI', 'SWHY']

FUTURES_CONFIG = {
    "SHSN300": {"symbol": "IF", "multiplier": 300, "name": "沪深300股指期货"},
    "SSE50": {"symbol": "IH", "multiplier": 300, "name": "上证50股指期货"},
    "SH000905": {"symbol": "IC", "multiplier": 200, "name": "中证500股指期货"},
    "CSI1000": {"symbol": "IM", "multiplier": 200, "name": "中证1000股指期货"},
    "SH932000": {"symbol": "IM(替代)", "multiplier": 200, "name": "中证2000(挂钩IM对冲)"}
}

DB_CONFIG = {
    "host": "192.168.116.107",
    "port": 5432,
    "database": "taichi_ops",
    "user": "postgres",
    "password": "123456",
    "timeout": 30
}
CONN_STR = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?connect_timeout={DB_CONFIG['timeout']}"

OVERWEIGHT_FACTOR = 1.2    # 允许个股最大超配基准的 120%
DELTA_SIZE = 0.02          # 市值因子最大绝对偏离 (Z-score)
DELTA_IND = 0.05           # BICS Level 2 全行业绝对偏离总和上限 (5%)


def parse_arguments():
    """解析命令行参数"""
    if len(sys.argv) < 4:
        print("\n" + "=" * 60)
        print("❌ 参数不足！使用方法：")
        print("   python run_custom_allocation_new.py <交易日期> <指数代码/名称> <初始Notional金额(元)>")
        print("\n📌 示例：")
        print("   python run_custom_allocation_new.py 20260915 000905 100000000")
        print("   python run_custom_allocation_new.py 20260915 CSI1000 50000000")
        print("   python run_custom_allocation_new.py 20260915 SH932000 50000000")
        print("=" * 60 + "\n")
        sys.exit(1)

    trade_date = re.sub(r'[^0-9]', '', str(sys.argv[1]))
    index_arg = str(sys.argv[2]).strip()

    notional_str = str(sys.argv[3]).strip().lower()
    if '亿' in notional_str or 'e8' in notional_str:
        notional = float(re.sub(r'[^\d.]', '', notional_str)) * 1e8
    elif '万' in notional_str or 'w' in notional_str or 'k' in notional_str:
        notional = float(re.sub(r'[^\d.]', '', notional_str)) * 1e4
    else:
        notional = float(re.sub(r'[^\d.]', '', notional_str))

    return trade_date, index_arg, notional


def get_index_spot_price(excel_file: pd.ExcelFile, target_sheet: str) -> float:
    """从 Index List Overview 概览页中读取指定指数最新点位"""
    overview_sheet = None
    for name in excel_file.sheet_names:
        if any(k in name.lower() for k in ['overview', 'list', 'summary', 'index']):
            overview_sheet = name
            break
    if not overview_sheet:
        overview_sheet = excel_file.sheet_names[0]

    try:
        df_overview = pd.read_excel(excel_file, sheet_name=overview_sheet, header=1)
        df_overview.columns = [str(c).strip() for c in df_overview.columns]

        ticker_col = next((c for c in df_overview.columns if 'ticker' in c.lower()), None)
        price_col = next((c for c in df_overview.columns if 'price' in c.lower() or 'value' in c.lower()), None)

        if not ticker_col or not price_col:
            raise ValueError(f"概览表 [{overview_sheet}] 中未找到 'Ticker' 或 'Value Price' 列")

        df_clean = df_overview.dropna(subset=[ticker_col, price_col]).copy()
        df_clean['Ticker_Clean'] = df_clean[ticker_col].astype(str).str.upper()

        target_key = target_sheet.upper().replace(" ", "")
        for _, row in df_clean.iterrows():
            ticker_val = row['Ticker_Clean'].replace(" ", "")
            if target_key in ticker_val or ticker_val.startswith(target_key):
                price = float(row[price_col])
                if price > 0:
                    return price

        clean_digits = re.sub(r'\D', '', target_key)
        if clean_digits:
            for _, row in df_clean.iterrows():
                if clean_digits in row['Ticker_Clean']:
                    price = float(row[price_col])
                    if price > 0:
                        return price

    except Exception as e:
        raise ValueError(f"❌ 读取概览表 [{overview_sheet}] 解析点位失败: {e}")

    raise ValueError(f"❌ 未能在概览表 [{overview_sheet}] 中找到指数 [{target_sheet}] 的点位记录！")


def calculate_futures_hedge_and_adjust_notional(matched_sheet: str, target_notional: float, index_price: float):
    """计算期货张数并对齐目标现货市值"""
    cfg = FUTURES_CONFIG.get(matched_sheet, {"symbol": "IM", "multiplier": 200, "name": "默认期货"})
    multiplier = cfg["multiplier"]
    symbol = cfg["symbol"]

    single_contract_val = float(index_price * multiplier)
    exact_contracts = target_notional / single_contract_val if single_contract_val > 0 else 0.0
    hedge_contracts = max(1, int(np.round(exact_contracts)))

    actual_stock_notional = float(hedge_contracts * multiplier * index_price)

    hedge_info = {
        "期货品种": symbol,
        "合约乘数": multiplier,
        "指数点位": round(float(index_price), 2),
        "单张合约价值(元)": round(single_contract_val, 2),
        "理论合约手数": round(float(exact_contracts), 2),
        "实际锁定手数(手)": hedge_contracts,
        "目标申报Notional(元)": round(float(target_notional), 2),
        "实际股票市值(元)": round(actual_stock_notional, 2),
        "对冲金额偏差(元)": round(actual_stock_notional - target_notional, 2)
    }

    return actual_stock_notional, hedge_info


def find_matched_index_df(indices_dict: dict, index_query: str):
    """匹配对应的成分表"""
    for k in indices_dict.keys():
        if index_query.upper() in k.upper():
            return k, indices_dict[k]
    clean_digits = re.sub(r'\D', '', index_query)
    if clean_digits:
        for k in indices_dict.keys():
            if clean_digits in k:
                return k, indices_dict[k]
    return None, None


def load_located_list(trade_date: str, candidate_dirs: list) -> pd.DataFrame:
    """加载锁券清单 Located List.xlsx"""
    located_file = None
    target_names = ["Located List.xlsx", "Located List.xls", f"Located_List_{trade_date}.xlsx"]

    search_paths = ["."] + candidate_dirs
    for sp in search_paths:
        if not os.path.exists(sp):
            continue
        for name in target_names:
            fp = os.path.join(sp, name)
            if os.path.isfile(fp):
                located_file = os.path.abspath(fp)
                break
        if located_file:
            break

    if not located_file:
        print("⚠️ 未找到 Located List.xlsx 文件，意向测算值将置为 0")
        return pd.DataFrame(columns=['stock_code', 'Located Qty', 'WA borrow cost'])

    print(f"📥 成功检测到并加载锁券清单: {located_file}")
    df_loc = pd.read_excel(located_file, dtype=str)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]

    code_col = next((c for c in df_loc.columns if '证券代码' in c), None)
    qty_col = next((c for c in df_loc.columns if '合约数量' in c), None)
    rate_col = next((c for c in df_loc.columns if any(k in c.lower() for k in ['利率', '费率', 'rate', 'cost'])), None)

    if not code_col or not qty_col:
        return pd.DataFrame(columns=['stock_code', 'Located Qty', 'WA borrow cost'])

    df_loc['stock_code'] = df_loc[code_col].astype(str).str.strip().str[:6].str.zfill(6)
    df_loc['qty'] = pd.to_numeric(df_loc[qty_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)

    if rate_col:
        clean_rate = df_loc[rate_col].astype(str).str.replace('%', '').str.replace(',', '').str.strip()
        df_loc['rate'] = pd.to_numeric(clean_rate, errors='coerce').fillna(0.0)
    else:
        df_loc['rate'] = 0.0

    df_loc['qty_x_rate'] = df_loc['qty'] * df_loc['rate']
    grouped = df_loc.dropna(subset=['stock_code']).groupby('stock_code', as_index=False).agg(
        total_qty=('qty', 'sum'),
        sum_qty_rate=('qty_x_rate', 'sum')
    )
    grouped['WA borrow cost'] = np.where(
        grouped['total_qty'] > 0,
        grouped['sum_qty_rate'] / grouped['total_qty'],
        0.0
    )
    return grouped[['stock_code', 'total_qty', 'WA borrow cost']].rename(columns={'total_qty': 'Located Qty'})


def fetch_stock_market_cap_from_db(stock_codes: list, trade_date: str) -> pd.DataFrame:
    """提取个股历史/最新市值"""
    engine = create_engine(CONN_STR)
    formatted_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
    codes_str = "','".join(stock_codes)

    query = text(f"""
        WITH ranked_cap AS (
            SELECT 
                code AS stock_code,
                market_cap,
                trade_date,
                ROW_NUMBER() OVER(PARTITION BY code ORDER BY trade_date DESC) as rn
            FROM public.md_stock_rk_info
            WHERE code IN ('{codes_str}')
              AND trade_date <= TO_DATE('{formatted_date}', 'YYYY-MM-DD')
              AND market_cap IS NOT NULL
        )
        SELECT stock_code, market_cap 
        FROM ranked_cap 
        WHERE rn = 1;
    """)

    with engine.connect() as conn:
        df_cap = pd.read_sql(query, conn)

    df_cap['stock_code'] = df_cap['stock_code'].astype(str).str.zfill(6)
    df_cap['market_cap'] = pd.to_numeric(df_cap['market_cap'], errors='coerce')
    return df_cap


def load_bics_industry_mapping(trade_date: str) -> pd.DataFrame:
    """加载本地 BICS 行业映射表"""
    ind_dir = "Industry Mappings"
    target_fp = os.path.join(ind_dir, f"industry_mapping_{trade_date}.xlsx")

    if not os.path.exists(target_fp):
        candidates = [os.path.join(ind_dir, f) for f in os.listdir(ind_dir) if f.startswith("industry_mapping") and f.endswith(".xlsx")]
        if not candidates:
            raise FileNotFoundError(f"❌ 未在 [{ind_dir}] 找到 industry_mapping 文件，请先运行行业拉取脚本！")
        target_fp = sorted(candidates)[-1]
        print(f"ℹ️ 自动加载最新 BICS 行业映射表: {os.path.basename(target_fp)}")
    else:
        print(f"📥 成功加载行业映射表: {os.path.basename(target_fp)}")

    df_ind = pd.read_excel(target_fp, dtype={'stock_code': str})
    df_ind['stock_code'] = df_ind['stock_code'].astype(str).str.zfill(6)
    df_ind['bics_level_2'] = df_ind['bics_level_2'].fillna('Others').astype(str).str.strip()
    return df_ind[['stock_code', 'bics_level_2']]


# ==============================================================================
# 🎯 2. 优化器核心与整手调整逻辑
# ==============================================================================
def apply_round_lot_with_residual_allocation(
    df: pd.DataFrame,
    adjusted_notional: float,
    avail_qty_col: str
) -> pd.DataFrame:
    """
    带整手残差资金再平衡的股数分配：
    1. 基础向下取整到 lot_size。
    2. 计算向下截断释放的资金总额，在富余券源的股票中按优化权重从大到小依次补齐 1 手，压低资金拖累。
    """
    df = df.copy()
    lot_size = df['lot_size'] if 'lot_size' in df.columns else 100.0

    # 1. 基础向下整手截断
    raw_qty = (df['opt_weight'] * adjusted_notional / df['price'])
    base_qty = np.floor(raw_qty / lot_size) * lot_size
    base_qty = np.minimum(base_qty, df[avail_qty_col].fillna(0.0))
    df['opt_qty'] = base_qty
    df['opt_mv'] = df['opt_qty'] * df['price']

    # 2. 残差分配 (若可用券源允许加 1 手且还有剩余现金)
    residual_cash = adjusted_notional - df['opt_mv'].sum()
    if residual_cash > 0:
        candidates = df.sort_values(by='opt_weight', ascending=False).index
        for idx in candidates:
            p = df.loc[idx, 'price']
            l_size = df.loc[idx, 'lot_size'] if 'lot_size' in df.columns else 100.0
            one_lot_val = p * l_size

            # 检查是否有资金且有富余券源满足 +1 手
            avail_q = df.loc[idx, avail_qty_col]
            curr_q = df.loc[idx, 'opt_qty']
            if residual_cash >= one_lot_val and (curr_q + l_size) <= avail_q:
                df.loc[idx, 'opt_qty'] += l_size
                df.loc[idx, 'opt_mv'] += one_lot_val
                residual_cash -= one_lot_val

    return df


def solve_hedging_portfolio_lp(
    df_model: pd.DataFrame,
    adjusted_notional: float,
    delta_size: float = DELTA_SIZE,
    delta_ind: float = DELTA_IND
):
    """多因子中性套利对冲优化求解器 (SciPy HiGHS LP Solver)"""
    df = df_model.copy().reset_index(drop=True)
    N = len(df)

    # 1. 市值因子 Z-Score
    ln_cap = np.log(np.maximum(df['market_cap'].values, 1.0))
    std_cap = ln_cap.std()
    df['size_zscore'] = (ln_cap - ln_cap.mean()) / (std_cap if std_cap > 0 else 1.0)

    # 2. 券源上限
    avail_qty_col = 'effective_avail_qty' if 'effective_avail_qty' in df.columns else 'internal_avail_qty'
    df['effective_avail_mv'] = df[avail_qty_col] * df['price']
    df['w_upper'] = np.minimum(df['weight'] * OVERWEIGHT_FACTOR, df['effective_avail_mv'] / adjusted_notional)
    df['w_upper'] = np.maximum(df['w_upper'].fillna(0.0), 0.0)

    # 3. 行业哑变量矩阵
    industries = sorted(df['bics_level_2'].dropna().unique())
    K = len(industries)
    ind_dummies = (
        pd.get_dummies(df['bics_level_2'], dtype=float)
        .reindex(columns=industries, fill_value=0.0)
        .values.astype(float)
    )

    w_bmk = df['weight'].values.astype(float)
    bmk_ind_weights = ind_dummies.T @ w_bmk
    bmk_size_exp = float(np.dot(w_bmk, df['size_zscore'].values))

    # 诊断
    total_upper = float(df['w_upper'].sum())
    zero_inv_count = int((df[avail_qty_col] == 0).sum())
    target_weight = min(1.0, total_upper)

    print("\n" + "=" * 65)
    print("🔬【券源与基准覆盖度可行域深度诊断】")
    print(f"📊 1. 总体券源体量:")
    print(f"   - 指数成分总数: {N} 支 | 完全无券股票: {zero_inv_count} 支 ({zero_inv_count/N*100:.1f}%)")
    print(f"   - 理论最高建仓总权重: {total_upper*100:.2f}% (本次目标权重: {target_weight*100:.2f}%)")

    ind_diagnostic = df.groupby('bics_level_2').agg(
        基准成分股数=('stock_code', 'count'),
        基准行业权重=('weight', lambda x: x.sum() * 100),
        有券股票数=(avail_qty_col, lambda x: (x > 0).sum()),
        行业可建仓上限=('w_upper', lambda x: x.sum() * 100)
    ).reset_index()

    ind_diagnostic['天然最小行业缺口(%)'] = np.maximum(0.0, ind_diagnostic['基准行业权重'] - ind_diagnostic['行业可建仓上限'])
    min_possible_ind_dev = ind_diagnostic['天然最小行业缺口(%)'].sum()

    print(f"\n📊 2. 行业偏离物理极限分析:")
    print(f"   - 若全部打满可用券源，全行业物理最小绝对偏离总和: 【 {min_possible_ind_dev:.2f}% 】")
    print(f"   - 模型硬约束预设上限: 【 {delta_ind*100:.2f}% 】")

    if min_possible_ind_dev > delta_ind * 100:
        print(f"🚨 诊断提示: 券源物理缺口 ({min_possible_ind_dev:.2f}%) 超过硬约束容差，将自动启用弹性软约束优化！")
    print("=" * 65 + "\n")

    # ==========================================================================
    # 🎯 Phase 1: 硬中性约束求解
    # ==========================================================================
    idx_w = 0
    idx_u = N
    idx_v = 2 * N
    num_vars_hard = 2 * N + K

    c_hard = np.zeros(num_vars_hard)
    c_hard[idx_u : idx_u + N] = 1.0  # min sum(u_i)

    bounds_hard = [(0.0, df.loc[i, 'w_upper']) for i in range(N)] + \
                  [(0.0, None) for _ in range(N)] + \
                  [(0.0, None) for _ in range(K)]

    A_ub_hard = []
    b_ub_hard = []

    # (1) 个股跟踪偏差: |w_i - w_bmk_i| <= u_i
    for i in range(N):
        r1 = np.zeros(num_vars_hard); r1[idx_w + i] = 1.0; r1[idx_u + i] = -1.0; A_ub_hard.append(r1); b_ub_hard.append(w_bmk[i])
        r2 = np.zeros(num_vars_hard); r2[idx_w + i] = -1.0; r2[idx_u + i] = -1.0; A_ub_hard.append(r2); b_ub_hard.append(-w_bmk[i])

    # (2) 行业偏差绝对值: |sum(X_ik * w_i) - bmk_ind_k| <= v_k
    for k in range(K):
        ind_col = ind_dummies[:, k]
        r1 = np.zeros(num_vars_hard); r1[idx_w : idx_w + N] = ind_col; r1[idx_v + k] = -1.0; A_ub_hard.append(r1); b_ub_hard.append(bmk_ind_weights[k])
        r2 = np.zeros(num_vars_hard); r2[idx_w : idx_w + N] = -ind_col; r2[idx_v + k] = -1.0; A_ub_hard.append(r2); b_ub_hard.append(-bmk_ind_weights[k])

    # 行业总偏差上限硬约束: sum(v_k) <= delta_ind
    r_ind = np.zeros(num_vars_hard); r_ind[idx_v : idx_v + K] = 1.0; A_ub_hard.append(r_ind); b_ub_hard.append(delta_ind)

    # (3) 市值因子偏离约束 (基于归一化敞口)
    size_vec = df['size_zscore'].values.astype(float)
    r_sz1 = np.zeros(num_vars_hard)
    r_sz1[idx_w: idx_w + N] = size_vec - (bmk_size_exp + delta_size)
    A_ub_hard.append(r_sz1)
    b_ub_hard.append(0.0)

    r_sz2 = np.zeros(num_vars_hard)
    r_sz2[idx_w: idx_w + N] = (bmk_size_exp - delta_size) - size_vec
    A_ub_hard.append(r_sz2)
    b_ub_hard.append(0.0)

    # (4) 组合建仓总权重区间约束 (下界弹性容差放宽至 5%，防止过紧导致无解)
    r_w1 = np.zeros(num_vars_hard); r_w1[idx_w : idx_w + N] = 1.0; A_ub_hard.append(r_w1); b_ub_hard.append(target_weight)
    r_w2 = np.zeros(num_vars_hard); r_w2[idx_w : idx_w + N] = -1.0; A_ub_hard.append(r_w2); b_ub_hard.append(-(target_weight * 0.95))

    res = linprog(c_hard, A_ub=np.array(A_ub_hard), b_ub=np.array(b_ub_hard), bounds=bounds_hard, method='highs')

    # ==========================================================================
    # 🎯 Phase 2: 弹性惩罚项兜底求解 (Elastic Penalty Formulation)
    # ==========================================================================
    if not res.success:
        print("⚠️ 券源存在物理行业断供，硬约束不可行，正在以【多因子弹性惩罚软约束】执行全局最优拟合...")

        idx_sp = 2 * N + K
        idx_sn = 2 * N + K + 1
        num_vars_soft = 2 * N + K + 2

        # 调高风格惩罚因子以匹配量纲
        LAMBDA_IND = 8.0     # 行业偏离惩罚
        LAMBDA_SIZE = 30.0   # 市值因子偏离惩罚 (增强市值控制力度)

        c_soft = np.zeros(num_vars_soft)
        c_soft[idx_u : idx_u + N] = 1.0
        c_soft[idx_v : idx_v + K] = LAMBDA_IND
        c_soft[idx_sp] = LAMBDA_SIZE
        c_soft[idx_sn] = LAMBDA_SIZE

        bounds_soft = [(0.0, df.loc[i, 'w_upper']) for i in range(N)] + \
                      [(0.0, None) for _ in range(N)] + \
                      [(0.0, None) for _ in range(K)] + \
                      [(0.0, None), (0.0, None)]

        A_ub_soft = []
        b_ub_soft = []

        for i in range(N):
            r1 = np.zeros(num_vars_soft); r1[idx_w + i] = 1.0; r1[idx_u + i] = -1.0; A_ub_soft.append(r1); b_ub_soft.append(w_bmk[i])
            r2 = np.zeros(num_vars_soft); r2[idx_w + i] = -1.0; r2[idx_u + i] = -1.0; A_ub_soft.append(r2); b_ub_soft.append(-w_bmk[i])

        for k in range(K):
            ind_col = ind_dummies[:, k]
            r1 = np.zeros(num_vars_soft); r1[idx_w : idx_w + N] = ind_col; r1[idx_v + k] = -1.0; A_ub_soft.append(r1); b_ub_soft.append(bmk_ind_weights[k])
            r2 = np.zeros(num_vars_soft); r2[idx_w : idx_w + N] = -ind_col; r2[idx_v + k] = -1.0; A_ub_soft.append(r2); b_ub_soft.append(-bmk_ind_weights[k])

        A_eq_soft = []
        b_eq_soft = []
        r_sz_eq = np.zeros(num_vars_soft)
        r_sz_eq[idx_w: idx_w + N] = size_vec - bmk_size_exp
        r_sz_eq[idx_sp] = -1.0
        r_sz_eq[idx_sn] = 1.0
        A_eq_soft.append(r_sz_eq)
        b_eq_soft.append(0.0)

        # 仓位总权重约束
        r_tot = np.zeros(num_vars_soft); r_tot[idx_w : idx_w + N] = 1.0
        A_ub_soft.append(r_tot); b_ub_soft.append(target_weight)
        r_tot_low = np.zeros(num_vars_soft); r_tot_low[idx_w : idx_w + N] = -1.0
        A_ub_soft.append(r_tot_low); b_ub_soft.append(-(target_weight * 0.90))

        res = linprog(
            c_soft,
            A_ub=np.array(A_ub_soft),
            b_ub=np.array(b_ub_soft),
            A_eq=np.array(A_eq_soft),
            b_eq=np.array(b_eq_soft),
            bounds=bounds_soft,
            method='highs'
        )

        if not res.success:
            print("⚠️ 进一步松弛总建仓下限，确保输出最优可行解...")
            A_ub_soft.pop()
            b_ub_soft.pop()
            res = linprog(
                c_soft,
                A_ub=np.array(A_ub_soft),
                b_ub=np.array(b_ub_soft),
                A_eq=np.array(A_eq_soft),
                b_eq=np.array(b_eq_soft),
                bounds=bounds_soft,
                method='highs'
            )

    if not res.success:
        raise RuntimeError(f"❌ 优化求解失败: {res.message}")

    # ==========================================================================
    # 🎯 提取权重、整手残差分配与风控指标计算
    # ==========================================================================
    opt_w = res.x[idx_w: idx_w + N]
    df['opt_weight'] = np.round(opt_w, 6)

    # 运用整手与零头资金再平衡分配逻辑
    df = apply_round_lot_with_residual_allocation(df, adjusted_notional, avail_qty_col)

    # 计算真实持仓归一化暴露
    tot_weight = df['opt_weight'].sum()
    if tot_weight > 0:
        opt_size_exp = float(np.dot(df['opt_weight'].values / tot_weight, df['size_zscore'].values))
    else:
        opt_size_exp = 0.0

    # 行业偏离计算
    opt_ind_weights = ind_dummies.T @ df['opt_weight'].values
    actual_ind_dev = float(np.sum(np.abs(opt_ind_weights - bmk_ind_weights)))

    print(f"✅ 优化求解成功！构建有效对冲现货 {int((df['opt_qty'] > 0).sum())} 支股票。")
    return df, opt_size_exp, bmk_size_exp, actual_ind_dev


# ==============================================================================
# 🎯 3. 主流程逻辑
# ==============================================================================
def run_custom_allocation():
    trade_date, index_arg, initial_notional = parse_arguments()

    # 1. 加载指数成分
    index_file = PathConfig.get_index_file(trade_date)
    if not os.path.exists(index_file):
        print(f"❌ 未找到指数文件: {index_file}")
        return

    excel_file = pd.ExcelFile(index_file)
    indices_dict = load_all_indices(index_file)
    matched_name, df_index = find_matched_index_df(indices_dict, index_arg)

    if df_index is None:
        print(f"❌ 未能匹配到指数 [{index_arg}]，支持的 Sheet 包括: {list(indices_dict.keys())}")
        return

    # 2. 期货对冲手数计算与市值规整
    spot_price = get_index_spot_price(excel_file, matched_name)
    adjusted_notional, hedge_info = calculate_futures_hedge_and_adjust_notional(
        matched_name, initial_notional, spot_price
    )

    print("=" * 65)
    print(f"🚀【多因子中性套利对冲优化系统 - LP求解模式】")
    print(f"📅 交易日期: [{trade_date}] | 标的指数: [{matched_name}] (成分股: {len(df_index)} 支)")
    print(f"📈 指数点位: {hedge_info['指数点位']:.2f} | 挂钩合约: [{hedge_info['期货品种']}] (乘数: {hedge_info['合约乘数']})")
    print(f"🎯 单张价值: {hedge_info['单张合约价值(元)'] / 1e4:.2f} 万元 ➔ 理论张数: {hedge_info['理论合约手数']} 手")
    print(f"⚡ 锁定对冲手数: 【 {hedge_info['实际锁定手数(手)']} 手 】")
    print(f"💰 初始需求 Notional: {initial_notional / 1e4:.2f} 万元")
    print(f"✅ 调整后目标现货市值: {adjusted_notional / 1e4:.2f} 万元 (偏差: {hedge_info['对冲金额偏差(元)'] / 1e4:+.2f} 万元)")
    print("=" * 65)

    # 3. 读取盘中券源 (内部底仓)
    source_dirs = PathConfig.get_source_dirs(trade_date)
    parser = HTIIntradayInventoryParser(exclude_sources=EXCLUDE_SOURCES)
    all_holdings = []

    for val_dir in source_dirs:
        if not os.path.exists(val_dir):
            continue
        files = [f for f in os.listdir(val_dir) if not f.startswith(('~', '.')) and f.lower().endswith(('.xlsx', '.xls'))]
        for filename in files:
            if "intraday" not in filename.lower() or trade_date not in filename:
                continue
            file_path = os.path.join(val_dir, filename)
            try:
                holdings = parser.parse(file_path, exclude_sources=EXCLUDE_SOURCES)
                all_holdings.extend(holdings)
                print(f"✅ 成功加载内部券源: {filename} ({len(holdings)} 条有效券源记录)")
            except Exception as e:
                print(f"❌ 读取券源文件失败 {filename}: {e}")

    if not all_holdings:
        print(f"❌ 未找到日期 [{trade_date}] 的有效 HTI_Intraday_Inventory 文件，流程终止。")
        return

    df_holdings = pd.DataFrame([h.__dict__ for h in all_holdings])
    df_holdings['stock_code'] = df_holdings['stock_code'].astype(str).str.zfill(6)
    df_holdings['quantity'] = pd.to_numeric(df_holdings['quantity'], errors='coerce').fillna(0.0)

    df_internal = df_holdings.groupby('stock_code', as_index=False)['quantity'].sum()
    df_internal.rename(columns={'quantity': 'internal_avail_qty'}, inplace=True)

    # 4. 数据合并与因子提取
    df_basket = df_index.copy()
    df_basket['stock_code'] = df_basket['stock_code'].astype(str).str.zfill(6)
    df_basket = df_basket.merge(df_internal, on='stock_code', how='left')
    df_basket['internal_avail_qty'] = df_basket['internal_avail_qty'].fillna(0.0)

    print("📡 正在从 DB 提取成分股历史/最新市值 (md_stock_rk_info)...")
    df_cap = fetch_stock_market_cap_from_db(df_basket['stock_code'].tolist(), trade_date)
    df_basket = df_basket.merge(df_cap, on='stock_code', how='left')
    df_basket['market_cap'] = df_basket['market_cap'].fillna(df_basket['market_cap'].median())

    df_ind = load_bics_industry_mapping(trade_date)
    df_basket = df_basket.merge(df_ind, on='stock_code', how='left')
    df_basket['bics_level_2'] = df_basket['bics_level_2'].fillna('Others')

    df_located = load_located_list(trade_date, candidate_dirs=source_dirs)
    df_basket = df_basket.merge(df_located, on='stock_code', how='left')
    df_basket['Located Qty'] = pd.to_numeric(df_basket['Located Qty'], errors='coerce').fillna(0.0)
    df_basket['WA borrow cost'] = pd.to_numeric(df_basket['WA borrow cost'], errors='coerce').fillna(0.0)

    df_basket['effective_avail_qty'] = df_basket['internal_avail_qty'] + df_basket['Located Qty']

    # 5. 求解多因子优化
    print("⚡ 正在通过多因子优化器求解最优现货空头对冲篮子...")
    df_opt, opt_size, bmk_size, ind_dev = solve_hedging_portfolio_lp(df_basket, adjusted_notional)

    # 6. 后续明细指标计算
    df_opt['internal_avail_qty'] = df_opt['internal_avail_qty'].fillna(0.0)
    df_opt['internal_avail_mv'] = df_opt['internal_avail_qty'] * df_opt['price']
    df_opt['effective_avail_mv'] = df_opt['effective_avail_qty'] * df_opt['price']

    lot_size = df_opt['lot_size'] if 'lot_size' in df_opt.columns else 100.0
    df_opt['target_theoretical_qty'] = np.floor((adjusted_notional * df_opt['weight'] / df_opt['price']) / lot_size) * lot_size

    df_opt['Located Qty'] = np.minimum(df_opt['Located Qty'], df_opt['target_theoretical_qty'])
    df_opt['Located MV'] = df_opt['Located Qty'] * df_opt['price']
    df_opt['Filled Weights'] = np.where(adjusted_notional > 0, df_opt['Located MV'] / adjusted_notional, 0.0)

    df_opt['shortage_qty'] = np.maximum(0.0, df_opt['target_theoretical_qty'] - df_opt['opt_qty'])
    df_opt['shortage_mv'] = df_opt['shortage_qty'] * df_opt['price']

    # 7. 看板数据汇总
    total_opt_mv = float(df_opt['opt_mv'].sum())
    total_opt_weight = float(df_opt['opt_weight'].sum())
    coverage_pct = round(total_opt_mv / adjusted_notional * 100.0, 2) if adjusted_notional > 0 else 0.0
    shortage_stock_count = int((df_opt['shortage_qty'] > 0).sum())

    total_located_mv = float(df_opt['Located MV'].sum())
    total_filled_weight = float(df_opt['Filled Weights'].sum())

    if total_located_mv > 0:
        portfolio_wa_borrow_cost = float((df_opt['WA borrow cost'] * df_opt['Located MV']).sum() / total_located_mv)
    else:
        portfolio_wa_borrow_cost = 0.0

    actual_used_located_qty = np.maximum(0.0, df_opt['opt_qty'] - df_opt['internal_avail_qty'])
    actual_used_located_mv = float((actual_used_located_qty * df_opt['price']).sum())
    actual_located_pct = (actual_used_located_mv / adjusted_notional * 100.0) if adjusted_notional > 0 else 0.0

    print(f"  🔹 外部借券真实消耗: {actual_used_located_mv / 1e4:.2f} 万元 | 占建仓比重: {actual_located_pct:.2f}% (内部底仓贡献: {100.0 - actual_located_pct:.2f}%)")

    # 主动偏离度（绝对与相对结构偏离）
    active_share_abs = 0.5 * np.sum(np.abs(df_opt['opt_weight'] - df_opt['weight']))
    w_norm = df_opt['opt_weight'] / total_opt_weight if total_opt_weight > 0 else df_opt['opt_weight']
    active_share_rel = 0.5 * np.sum(np.abs(w_norm - df_opt['weight']))

    summary_dashboard = pd.DataFrame([{
        "指数代码标识": matched_name,
        "挂钩期货合约": hedge_info["期货品种"],
        "合约乘数": hedge_info["合约乘数"],
        "指数估值点位": hedge_info["指数点位"],
        "期货锁定手数(手)": hedge_info["实际锁定手数(手)"],
        "调整后对冲目标市值(元)": adjusted_notional,
        "算法优化实际建仓市值(元)": round(total_opt_mv, 2),
        "算法实际市值覆盖率(%)": coverage_pct,
        "优化组合总权重(%)": round(total_opt_weight * 100.0, 4),
        "组合市值因子暴露": round(opt_size, 4),
        "基准市值因子暴露": round(bmk_size, 4),
        "市值因子偏离(Z-score)": round(abs(opt_size - bmk_size), 4),
        "BICS二级全行业总绝对偏离(%)": round(ind_dev * 100.0, 4),
        "绝对主动偏离度ActiveShare(%)": round(active_share_abs * 100.0, 2),
        "相对结构偏离度RelativeActiveShare(%)": round(active_share_rel * 100.0, 2),
        "前期意向Located总市值(元)": round(total_located_mv, 2),
        "意向Filled权重合计(%)": round(total_filled_weight * 100.0, 4),
        "意向加权借券费率(%)": round(portfolio_wa_borrow_cost, 4),
        "完全配置股票数": int((df_opt['opt_qty'] > 0).sum()),
        "存在代偿缺口股票数": shortage_stock_count,
        "指数总成分数": len(df_opt)
    }])

    # 8. 导出清单
    export_cols = [
        'stock_code', 'stock_name', 'bics_level_2', 'price', 'weight',
        'opt_weight', 'opt_qty', 'opt_mv',
        'target_theoretical_qty', 'shortage_qty', 'shortage_mv',
        'internal_avail_qty', 'internal_avail_mv',
        'Located Qty', 'Located MV', 'effective_avail_qty', 'effective_avail_mv',
        'Filled Weights', 'WA borrow cost'
    ]
    df_export = df_opt[export_cols].copy()
    df_export.rename(columns={
        'stock_code': '股票代码',
        'stock_name': '股票名称',
        'bics_level_2': 'BICS二级行业',
        'price': '最新价格',
        'weight': '基准权重',
        'opt_weight': '优化拟合权重',
        'opt_qty': '优化分配股数',
        'opt_mv': '优化分配市值',
        'target_theoretical_qty': '理论完全复制股数',
        'shortage_qty': '代偿缺口股数',
        'shortage_mv': '缺口市值',
        'internal_avail_qty': '内部底仓可用股数',
        'internal_avail_mv': '内部底仓可用市值',
        'Located Qty': '意向Located股数',
        'Located MV': '意向Located市值',
        'effective_avail_qty': '总有效可用股数(内部+已锁)',
        'effective_avail_mv': '总有效可用市值',
        'Filled Weights': '意向Filled权重',
        'WA borrow cost': '意向借券费率(%)'
    }, inplace=True)

    output_filename = f"./Allocation_LP_{matched_name}_{hedge_info['实际锁定手数(手)']}Lot_{int(adjusted_notional / 1e4)}W_{trade_date}.xlsx"
    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
        summary_dashboard.to_excel(writer, sheet_name="对冲与多因子风控看板", index=False)
        df_export.to_excel(writer, sheet_name="优化成分股明细", index=False)

    # 9. 控制台诊断展示
    overweight_mask = df_opt['opt_weight'] > (df_opt['weight'] + 1e-6)
    zero_alloc_mask = (df_opt['opt_qty'] == 0) & (df_opt['weight'] > 0)
    capped_mask = df_opt['opt_weight'] >= (df_opt['weight'] * (OVERWEIGHT_FACTOR - 0.005))
    overweight_mv = (df_opt.loc[overweight_mask, 'opt_mv'] - (df_opt.loc[overweight_mask, 'weight'] * adjusted_notional)).sum()

    print("\n" + "=" * 65)
    print("🧬【模型替代 / 代偿深度分析】")
    print(f"  🔹 绝对主动偏离度 (Active Share): {active_share_abs * 100:.2f}% | 纯结构偏离: {active_share_rel * 100:.2f}%")
    print(f"  🔹 超配代偿股票数: {overweight_mask.sum()} 支 | 顶格超配(1.2倍)股票数: {capped_mask.sum()} 支")
    print(f"  🔹 净超配代偿市值: {overweight_mv / 1e4:.2f} 万元 (占总建仓市值的 {overweight_mv / total_opt_mv * 100:.2f}%)")
    print(f"  🔹 完全被抛弃(0建仓)成分股: {zero_alloc_mask.sum()} 支 (权重占比: {df_opt.loc[zero_alloc_mask, 'weight'].sum() * 100:.2f}%)")
    print("=" * 65)

    print("\n" + "=" * 65)
    print("📊【券源匹配与多因子对冲优化总览】")
    print(f"  🔹 对应期货: {hedge_info['期货品种']} | 锁定建仓: {hedge_info['实际锁定手数(手)']} 手")
    print(f"  🔹 目标股票市值: {adjusted_notional / 1e4:.2f} 万元 | 优化实际建仓: {total_opt_mv / 1e4:.2f} 万元 (覆盖率 {coverage_pct}%)")
    print(f"  🔹 市值中性检验: 组合 {opt_size:.4f} vs 基准 {bmk_size:.4f} (偏离: {abs(opt_size - bmk_size):.4f} <= {DELTA_SIZE})")
    print(f"  🔹 行业中性检验: BICS Level 2 全行业总偏离: {ind_dev * 100:.2f}% (<= {DELTA_IND * 100:.1f}%)")
    print(f"  🔹 前期 Located 意向市值: {total_located_mv / 1e4:.2f} 万元 | 意向 Filled 权重: {total_filled_weight * 100:.2f}%")
    print(f"\n🎉 专属多因子优化分配清单已生成: {output_filename}")
    print("=" * 65 + "\n")

    generate_dual_locate_demand_lists(
        df_opt=df_opt,
        trade_date=trade_date,
        matched_name=matched_name,
        adjusted_notional=adjusted_notional
    )


if __name__ == "__main__":
    run_custom_allocation()