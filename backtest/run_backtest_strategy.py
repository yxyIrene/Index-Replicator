import os
import sys
import re
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Run Example
# python backtest/run_backtest_strategy.py Strat_1 20260801 20260930 daily
# python backtest/run_backtest_strategy.py Strat_1 20260801 20260930 weekly
# python backtest/run_backtest_strategy.py Strat_2 20260801 20260930 daily
# =========================================================================
# 0. 自动将项目根目录加入 sys.path，确保能顺利导入 config 与 replication
# =========================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    CONN_STR, RESTRICTION_SHARE_DIR, FUTURES_CONFIG,
    OVERWEIGHT_FACTOR, DELTA_SIZE, DELTA_IND, EXCLUDE_SOURCES
)
from replication.date_utils import PathConfig
from replication.index_loader import (
    load_all_indices_from_db, get_index_spot_price_from_db, INDEX_DB_MAPPING
)
from replication.data_loaders import (
    fetch_stock_market_cap_from_db, fetch_st_stocks_from_db,
    load_jump_restriction_list, load_bics_industry_mapping,
    load_located_list, load_internal_inventory
)
from replication.hedge_calculator import calculate_futures_hedge_and_adjust_notional
from replication.solvers import solve_hedging_portfolio


# =========================================================================
# 1. 数据库数据拉取
# =========================================================================
def get_trading_days(conn_str: str, start_date: str, end_date: str) -> List[str]:
    """从数据库获取历史有效交易日列表 (升序)"""
    engine = create_engine(conn_str)
    s_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    e_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

    query = text(f"""
        SELECT DISTINCT trade_date 
        FROM public.md_index_daily 
        WHERE trade_date >= TO_DATE('{s_date}', 'YYYY-MM-DD')
          AND trade_date <= TO_DATE('{e_date}', 'YYYY-MM-DD')
        ORDER BY trade_date ASC;
    """)
    with engine.connect() as conn:
        df_days = pd.read_sql(query, conn)

    return [d.strftime('%Y%m%d') for d in pd.to_datetime(df_days['trade_date'])]


def get_market_eod_data(
    conn_str: str, trade_date: str, benchmark: str
) -> tuple[dict, float]:
  """获取指定交易日全市场股票最新收盘价和基准指数收盘点位（带有效值向前回溯兜底）"""
  engine = create_engine(conn_str)
  fmt_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
  daily_code = INDEX_DB_MAPPING.get(benchmark.upper(), {}).get(
      'daily_code', '000852'
  )

  # 🎯 改为取 <= trade_date 的最近一个有效点位，绝不退化为 1.0
  query_bmk = text(f"""
        SELECT close_price FROM public.md_index_daily 
        WHERE code = '{daily_code}' 
          AND trade_date <= TO_DATE('{fmt_date}', 'YYYY-MM-DD')
          AND close_price IS NOT NULL
        ORDER BY trade_date DESC
        LIMIT 1;
    """)

  # 股票行情：优先取当天，若当天无记录则往前取最近一个有效收盘价
  query_stocks = text(f"""
        SELECT code, COALESCE(close_price, pre_close) as price 
        FROM public.md_stock_daily 
        WHERE trade_date = TO_DATE('{fmt_date}', 'YYYY-MM-DD');
    """)

  with engine.connect() as conn:
    df_bmk = pd.read_sql(query_bmk, conn)
    df_stocks = pd.read_sql(query_stocks, conn)

  if not df_bmk.empty and pd.notna(df_bmk['close_price'].iloc[0]):
    bmk_spot = float(df_bmk['close_price'].iloc[0])
  else:
    bmk_spot = None  # 缺失明确置 None，交由收益率计算层处理

  df_stocks['stock_code'] = (
      df_stocks['code'].astype(str).str.extract(r'(\d{6})')[0].str.zfill(6)
  )
  df_stocks = df_stocks.dropna(subset=['price'])
  stock_prices = dict(zip(df_stocks['stock_code'], df_stocks['price']))

  return stock_prices, bmk_spot


def is_rebalance_day(current_idx: int, t_date: str, next_date: str, rebal_freq: Union[str, int]) -> bool:
    """根据频率配置判断是否再平衡"""
    if str(rebal_freq).lower() == "daily":
        return True

    if isinstance(rebal_freq, int) or str(rebal_freq).isdigit():
        step = int(rebal_freq)
        return current_idx % step == 0

    curr_dt = pd.to_datetime(t_date)
    next_dt = pd.to_datetime(next_date) if next_date else None

    if str(rebal_freq).lower() == "weekly":
        if next_dt is None or next_dt.isocalendar().week != curr_dt.isocalendar().week:
            return True
        return False

    if str(rebal_freq).lower() == "monthly":
        if next_dt is None or next_dt.month != curr_dt.month:
            return True
        return False

    return True


# =========================================================================
# 2. 策略回测主逻辑
# =========================================================================
def run_strategy_backtest(
        strategy_id: str,
        start_date: str,
        end_date: str,
        rebal_freq: Union[str, int] = "daily",
        config_path: str = None
):
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config", "Strategy.xlsx")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 找不到策略配置文件: {config_path}")

    # 读取策略参数
    df_cfg = pd.read_excel(config_path)
    col_map = {str(c).strip().lower(): str(c).strip() for c in df_cfg.columns}
    id_col = col_map.get('strategy_id', 'Strategy_ID')

    strat_rows = df_cfg[df_cfg[id_col].astype(str).str.strip().str.upper() == strategy_id.strip().upper()]
    if strat_rows.empty:
        raise ValueError(f"❌ 在 {config_path} 中未找到策略 [{strategy_id}]！可选ID: {df_cfg[id_col].tolist()}")

    row = strat_rows.iloc[0]

    def get_cfg_val(k, def_val):
        real_k = col_map.get(k.lower())
        return def_val if not real_k or pd.isna(row[real_k]) else row[real_k]

    benchmark = str(get_cfg_val('Benchmark', 'CSI1000')).strip()
    n_raw = str(get_cfg_val('Notional', '50000000')).strip().lower()
    initial_notional = float(re.sub(r'[^\d.]', '', n_raw)) * (
        1e8 if '亿' in n_raw or 'e8' in n_raw else (1e4 if any(x in n_raw for x in ['万', 'w', 'k']) else 1.0))
    solver_type = str(get_cfg_val('Solver_Type', 'LP')).strip().upper()
    overweight = float(get_cfg_val('Overweight_Factor', OVERWEIGHT_FACTOR))
    delta_sz = float(get_cfg_val('Delta_Size', DELTA_SIZE))
    delta_in = float(get_cfg_val('Delta_Ind', DELTA_IND))
    exclude_st_flag = bool(get_cfg_val('Exclude_ST', True))
    exclude_res_flag = bool(get_cfg_val('Exclude_Restriction', True))
    lambda_ind = float(get_cfg_val('Lambda_Ind', 8.0))
    lambda_size = float(get_cfg_val('Lambda_Size', 30.0))

    print("\n" + "=" * 70)
    print(f"🎯【回测配置】策略: [{strategy_id}] | 基准: [{benchmark}] | 调仓机制: [{rebal_freq}]")
    print(
        f"   规模: {initial_notional / 1e4:.0f} 万元 | 引擎: [{solver_type}] | 超配: {overweight}x | Size差: {delta_sz} | Ind差: {delta_in * 100:.1f}%")
    print("=" * 70)

    trading_days = get_trading_days(CONN_STR, start_date, end_date)
    print(f"📅 回测区间: {start_date} -> {end_date} (共 {len(trading_days)} 个交易日)")
    if len(trading_days) < 2:
        print("❌ 交易日天数过少，至少需要 2 个交易日！")
        return

    current_holdings_qty = {}
    records = []

    for i in range(len(trading_days) - 1):
        t_date = trading_days[i]
        next_date = trading_days[i + 1]
        rebal_flag = is_rebalance_day(i, t_date, next_date, rebal_freq)

        # 调仓日或首次持仓为空 -> 运行优化器
        if rebal_flag or not current_holdings_qty:
            print(f"🔄 [{i + 1}/{len(trading_days) - 1}] 日期 {t_date}: 触发再平衡优化 (Rebalance)...")
            indices_dict = load_all_indices_from_db(CONN_STR, t_date)
            matched_name = next((k for k in indices_dict if benchmark.upper() in k.upper()), None)

            if matched_name:
                df_basket = indices_dict[matched_name].copy()
                spot_price = get_index_spot_price_from_db(CONN_STR, matched_name, t_date)
                adjusted_notional, hedge_info = calculate_futures_hedge_and_adjust_notional(
                    matched_name, initial_notional, spot_price, FUTURES_CONFIG
                )

                # 底仓与锁券
                source_dirs = PathConfig.get_source_dirs(t_date)
                try:
                    df_basket = df_basket.merge(load_internal_inventory(source_dirs, t_date, EXCLUDE_SOURCES),
                                                on='stock_code', how='left')
                except Exception:
                    df_basket['internal_avail_qty'] = 0.0
                df_basket['internal_avail_qty'] = df_basket['internal_avail_qty'].fillna(0.0)

                # 市值与行业
                df_cap = fetch_stock_market_cap_from_db(CONN_STR, df_basket['stock_code'].tolist(), t_date)
                df_basket = df_basket.merge(df_cap, on='stock_code', how='left')
                df_basket['market_cap'] = df_basket['market_cap'].fillna(df_basket['market_cap'].median())

                df_basket = df_basket.merge(load_bics_industry_mapping(t_date), on='stock_code', how='left')
                df_basket['bics_level_2'] = df_basket['bics_level_2'].fillna('Others')

                try:
                    df_basket = df_basket.merge(load_located_list(t_date, source_dirs), on='stock_code', how='left')
                except Exception:
                    df_basket['Located Qty'] = 0.0
                df_basket['Located Qty'] = pd.to_numeric(df_basket.get('Located Qty', 0.0), errors='coerce').fillna(0.0)
                df_basket['effective_avail_qty'] = df_basket['internal_avail_qty'] + df_basket['Located Qty']

                # 风控排除
                exclude_pool = set()
                if exclude_st_flag:
                    exclude_pool |= fetch_st_stocks_from_db(CONN_STR, t_date)
                if exclude_res_flag:
                    exclude_pool |= load_jump_restriction_list(RESTRICTION_SHARE_DIR, t_date)

                mask_exc = df_basket['stock_code'].isin(exclude_pool)
                df_basket.loc[mask_exc, ['internal_avail_qty', 'Located Qty', 'effective_avail_qty']] = 0.0

                # 优化求解
                df_opt, opt_size, bmk_size, ind_dev = solve_hedging_portfolio(
                    df_model=df_basket, adjusted_notional=adjusted_notional, solver_type=solver_type,
                    delta_size=delta_sz, delta_ind=delta_in, overweight_factor=overweight,
                    enable_elastic=True, lambda_ind=lambda_ind, lambda_size=lambda_size
                )

                valid_holds = df_opt[df_opt['opt_qty'] > 0]
                current_holdings_qty = dict(zip(valid_holds['stock_code'], valid_holds['opt_qty']))

        # =============================================================
        # B. 结算 t -> t+1 日的表现 (自然漂移收益计算与估值核算)
        # =============================================================
        # 1. 获取 t 日与 t+1 日的个股收盘价字典及基准收盘点位
        prices_t, bmk_spot_t = get_market_eod_data(CONN_STR, t_date, benchmark)
        prices_t1, bmk_spot_t1 = get_market_eod_data(CONN_STR, next_date, benchmark)

        # 2. 🎯 核心估值计算：持仓股数 (Qty) × 当日收盘价 (Price)
        # t 日收盘持仓估值:
        mv_t = sum(qty * prices_t.get(code, 0.0) for code, qty in current_holdings_qty.items())

        # t+1 日收盘持仓估值 (若 t+1 日某股票停牌无行情，则沿用 t 日价格 prices_t.get(code, 0.0)):
        mv_t1 = sum(
            qty * prices_t1.get(code, prices_t.get(code, 0.0)) for code, qty in current_holdings_qty.items())

        # 3. 计算收益率
        port_ret = (mv_t1 / mv_t - 1.0) if mv_t > 0 else 0.0
        bmk_ret = (bmk_spot_t1 / bmk_spot_t - 1.0) if bmk_spot_t > 0 else 0.0
        excess_ret = port_ret - bmk_ret
        records.append({
            "Trade_Date": t_date,
            "Next_Date": next_date,
            "Is_Rebal_Day": rebal_flag,
            "Holdings_Count": len(current_holdings_qty),
            "Holdings_MV": round(mv_t, 2),
            "Port_Return(%)": round(port_ret * 100, 4),
            "Bmk_Return(%)": round(bmk_ret * 100, 4),
            "Excess_Return(%)": round(excess_ret * 100, 4)
        })

    # 输出绩效评价指标
    df_res = pd.DataFrame(records)
    df_res['NAV_Port'] = (1.0 + df_res['Port_Return(%)'] / 100.0).cumprod()
    df_res['NAV_Bmk'] = (1.0 + df_res['Bmk_Return(%)'] / 100.0).cumprod()
    df_res['NAV_Excess'] = df_res['NAV_Port'] / df_res['NAV_Bmk']

    daily_excess = df_res['Excess_Return(%)'] / 100.0
    tracking_error = daily_excess.std() * np.sqrt(252)
    mean_excess_annual = daily_excess.mean() * 252
    info_ratio = mean_excess_annual / tracking_error if tracking_error > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"🏁【回测绩效看板: {strategy_id}】({start_date} -> {end_date} | 调仓: {rebal_freq})")
    print("=" * 70)
    print(f"  🔹 组合累计收益:     {(df_res['NAV_Port'].iloc[-1] - 1.0) * 100:.2f}%")
    print(f"  🔹 基准累计收益:     {(df_res['NAV_Bmk'].iloc[-1] - 1.0) * 100:.2f}%")
    print(f"  🔹 累计超额偏离:     {(df_res['NAV_Excess'].iloc[-1] - 1.0) * 100:+.2f}%")
    print(f"  🔹 年化跟踪误差(TE): {tracking_error * 100:.2f}%")
    print(f"  🔹 年化信息比率(IR): {info_ratio:.2f}")
    print(f"  🔹 平均持仓只数:     {df_res['Holdings_Count'].mean():.0f} 支")
    print("=" * 70)

    # 导出结果文件至 backtest/ 目录下
    out_file = os.path.join(CURRENT_DIR,
                            f"Backtest_{strategy_id}_{benchmark}_{rebal_freq}_{start_date}_{end_date}.xlsx")
    df_res.to_excel(out_file, index=False)
    print(f"📁 详细结果已导出至: {out_file}\n")


if __name__ == "__main__":
    # 支持命令行参数: python backtest/run_backtest_strategy.py <Strategy_ID> <起始日> <结束日> [调仓频率]
    # 示例: python backtest/run_backtest_strategy.py Strat_1 20260801 20260930 daily
    strat_arg = sys.argv[1] if len(sys.argv) >= 2 else "Strat_1"
    start_arg = sys.argv[2] if len(sys.argv) >= 3 else "20260801"
    end_arg = sys.argv[3] if len(sys.argv) >= 4 else "20260930"
    freq_arg = sys.argv[4] if len(sys.argv) >= 5 else "daily"

    run_strategy_backtest(
        strategy_id=strat_arg,
        start_date=start_arg,
        end_date=end_arg,
        rebal_freq=freq_arg
    )