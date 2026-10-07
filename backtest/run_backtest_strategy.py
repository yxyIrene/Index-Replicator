import os
import sys
import re
import warnings
from typing import Dict, List, Union, Tuple, Optional
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


# 运行示例:
# python backtest/run_backtest_strategy.py Strategy_CSI1000_QP 20260803 20260930 weekly
# python backtest/run_backtest_strategy.py Strategy_CSI1000_LP 20260803 20260930 daily

# =========================================================================
# 0. 环境与路径配置
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
    load_located_list, load_internal_inventory,
    get_engine,
    fetch_suspension_stocks_from_db
)
from replication.hedge_calculator import calculate_futures_hedge_and_adjust_notional
from replication.solvers import solve_hedging_portfolio

FIELD_DEFINITIONS = [
    # Daily_TimeSeries
    {"归属表格": "Daily_TimeSeries", "字段列名": "Trade_Date", "中文释义": "交易日期", "业务说明与计算公式": "当前持仓估值与收益结算的实际交易日 (YYYYMMDD)"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Is_Rebal_Day", "中文释义": "再平衡调仓标记", "业务说明与计算公式": "1 表示当日执行优化器调仓，0 表示持仓股数恒定自然漂移"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Index_Spot_Price", "中文释义": "指数现货点位", "业务说明与计算公式": "基准指数官方收盘点位"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Index_Return(%)", "中文释义": "基准单日涨跌幅(%)", "业务说明与计算公式": "(Index_t / Index_t-1 - 1) * 100，首日为 0"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Index_NAV", "中文释义": "基准指数累计净值", "业务说明与计算公式": "(1 + Index_Return).cumprod()，以 1.0 为起点"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Stock_Basket_NAV", "中文释义": "股票篮子现货净值", "业务说明与计算公式": "(1 + Stock_Basket_Return).cumprod()，**与 Index_NAV 重叠对比的核心曲线**"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Short_NAV", "中文释义": "融券空头组合净值", "业务说明与计算公式": "(1 - Stock_Basket_Return).cumprod()，融券空头真实净值曲线"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Hedged_NAV", "中文释义": "期现对冲累计净偏离曲线", "业务说明与计算公式": "(1 + Net_Tracking_Return).cumprod()，抹平大盘后的对冲纯超额曲线"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Futures_Symbol", "中文释义": "对冲期货合约品种", "业务说明与计算公式": "IM (中证1000)、IC (中证500)、IF (沪深300)"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Futures_Multiplier", "中文释义": "期货合约乘数", "业务说明与计算公式": "IM为200元/点，IC为200元/点，IF为300元/点"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Futures_Contracts", "中文释义": "期货多头锁定手数", "业务说明与计算公式": "开仓日根据本金锁定并全程保持不变的对冲合约张数"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Futures_Notional_MV", "中文释义": "期货多头对应指数市值", "业务说明与计算公式": "Futures_Contracts * Futures_Multiplier * Index_Spot_Price"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Futures_PnL", "中文释义": "期货多头单日盯市盈亏(元)", "业务说明与计算公式": "Contracts * Multiplier * (Index_t - Index_t-1)，首日为 0"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Futures_Cum_PnL", "中文释义": "期货多头累计盯市盈亏(元)", "业务说明与计算公式": "Futures_PnL 逐日累加求和"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Holdings_Count", "中文释义": "融券空头持仓股票只数", "业务说明与计算公式": "经优化器求解后持仓股数 > 0 的有效股票总数"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Short_Equity_MV", "中文释义": "融券空头股票篮子市值(元)", "业务说明与计算公式": "sum(持仓股数 Qty_i * 个股收盘价 Price_i,t)"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Short_Equity_PnL", "中文释义": "融券空头单日盯市盈亏(元)", "业务说明与计算公式": "-sum(Qty_i * (Price_i,t - Price_i,t-1))，股票跌为正收益，首日为 0"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Short_Equity_Cum_PnL", "中文释义": "融券空头累计盈亏(元)", "业务说明与计算公式": "Short_Equity_PnL 逐日累加求和"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Stock_Basket_Return(%)", "中文释义": "股票篮子现货单日涨跌幅(%)", "业务说明与计算公式": "股票组合多头视角的涨跌百分比，用于对比基准指数涨跌"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "MV_Diff", "中文释义": "期现名义市值偏离(元)", "业务说明与计算公式": "Futures_Notional_MV - Short_Equity_MV，衡量期现市值匹配精度"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Net_Tracking_PnL", "中文释义": "期现对冲单日净偏离损益(元)", "业务说明与计算公式": "Futures_PnL + Short_Equity_PnL (理想完全复制状态下严格为0)"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Net_Tracking_Cum_PnL", "中文释义": "期现对冲累计净偏离损益(元)", "业务说明与计算公式": "Net_Tracking_PnL 逐日累加求和"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Net_Tracking_Return(%)", "中文释义": "单日对冲净收益率(%)", "业务说明与计算公式": "(Net_Tracking_PnL / Futures_Notional_MV_prev) * 100"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "NAV_Peak", "中文释义": "对冲净偏离历史最高值", "业务说明与计算公式": "Hedged_NAV.cummax()，用于核算回撤"},
    {"归属表格": "Daily_TimeSeries", "字段列名": "Drawdown(%)", "中文释义": "期现对冲最大回撤百分比(%)", "业务说明与计算公式": "(Hedged_NAV - NAV_Peak) / NAV_Peak * 100"},

    # Performance_Summary
    {"归属表格": "Performance_Summary", "字段列名": "年化跟踪误差 (TE)", "中文释义": "Annualized Tracking Error", "业务说明与计算公式": "std(Net_Tracking_Return) * sqrt(252)，纯粹衡量优化器与券源的拟合质量"},
    {"归属表格": "Performance_Summary", "字段列名": "年化信息比率 (IR)", "中文释义": "Annualized Information Ratio", "业务说明与计算公式": "年化对冲超额收益 / 年化跟踪误差"},
    {"归属表格": "Performance_Summary", "字段列名": "年化夏普比率 (Sharpe Ratio)", "中文释义": "Annualized Sharpe Ratio", "业务说明与计算公式": "年化对冲超额收益 / 年化波动率"},
    {"归属表格": "Performance_Summary", "字段列名": "卡玛比率 (Calmar Ratio)", "中文释义": "Calmar Ratio", "业务说明与计算公式": "年化对冲超额收益 / 最大回撤绝对值"},
    {"归属表格": "Performance_Summary", "字段列名": "最大回撤 (Max Drawdown)", "中文释义": "对冲净值最大回撤深度", "业务说明与计算公式": "min(Drawdown(%))"},
    {"归属表格": "Performance_Summary", "字段列名": "期现名义市值平均偏离", "中文释义": "Mean Absolute MV Gap", "业务说明与计算公式": "mean(|Futures_Notional_MV - Short_Equity_MV|)"},

    # Hedge_Params_Log
    {"归属表格": "Hedge_Params_Log", "字段列名": "Rebal_Date", "中文释义": "调仓日期", "业务说明与计算公式": "触发再平衡优化的交易日"},
    {"归属表格": "Hedge_Params_Log", "字段列名": "Locked_Contracts", "中文释义": "锁定期货合约手数", "业务说明与计算公式": "建仓日根据 5000 万名义本金锁定的期货手数"},
    {"归属表格": "Hedge_Params_Log", "字段列名": "Target_Stock_Notional", "中文释义": "动态跟随目标市值(元)", "业务说明与计算公式": "Locked_Contracts * Multiplier * Spot_Price_t，输入优化器的目标市值"},
    {"归属表格": "Hedge_Params_Log", "字段列名": "Is_Initial_Open", "中文释义": "是否为首次建仓日", "业务说明与计算公式": "1 表示首次开仓，0 表示常规再平衡"}
]

# === 1. 数据库数据提取服务 ===
def get_trading_days(conn_str: str, start_date: str, end_date: str) -> List[str]:
    """从 TAICHI 权威交易日历表 md_work_day_new 精确提取中国 A 股 (SSE) 交易日"""
    engine = get_engine(conn_str)
    s_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    e_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

    query = text("""
        SELECT DISTINCT trade_date 
        FROM public.md_work_day_new 
        WHERE trade_date >= :s_date
          AND trade_date <= :e_date
          AND exchange = 'SSE'
          AND is_trade_day = true
        ORDER BY trade_date ASC;
    """)

    with engine.connect() as conn:
        df_days = pd.read_sql(query, conn, params={"s_date": s_date, "e_date": e_date})

    if df_days.empty:
        raise ValueError(f"❌ 在区间 [{start_date} ~ {end_date}] 内未查到任何 A 股交易日，请检查数据库日历！")

    return [d.strftime('%Y%m%d') for d in pd.to_datetime(df_days['trade_date'])]


def get_market_eod_data(
    conn_str: str, trade_date: str, benchmark: str
) -> Tuple[Dict[str, float], Optional[float], set]:
    """
    获取指定交易日全市场股票最新收盘价、基准指数收盘点位、当日停牌股票集合。
    返回: (stock_prices, bmk_spot, suspended_stocks)
    """
    engine = get_engine(conn_str)
    fmt_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
    daily_code = INDEX_DB_MAPPING.get(benchmark.upper(), {}).get('daily_code', '000852')

    query_bmk = text("""
        SELECT close_price FROM public.md_index_daily 
        WHERE code = :daily_code 
          AND trade_date <= :fmt_date
          AND close_price IS NOT NULL
        ORDER BY trade_date DESC 
        LIMIT 1;
    """)

    query_stocks = text("""
        SELECT code, COALESCE(close_price, pre_close) as price 
        FROM public.md_stock_daily 
        WHERE trade_date = :fmt_date;
    """)

    with engine.connect() as conn:
        df_bmk = pd.read_sql(query_bmk, conn, params={"daily_code": daily_code, "fmt_date": fmt_date})
        df_stocks = pd.read_sql(query_stocks, conn, params={"fmt_date": fmt_date})

    bmk_spot = float(df_bmk['close_price'].iloc[0]) if (not df_bmk.empty and pd.notna(df_bmk['close_price'].iloc[0])) else None

    df_stocks['stock_code'] = (
        df_stocks['code'].astype(str).str.extract(r'(\d{6})')[0].str.zfill(6)
    )
    df_stocks = df_stocks.dropna(subset=['price'])
    df_stocks = df_stocks[df_stocks['price'] > 0.01]
    stock_prices = dict(zip(df_stocks['stock_code'], df_stocks['price']))

    suspended_stocks = fetch_suspension_stocks_from_db(conn_str, trade_date)

    return stock_prices, bmk_spot, suspended_stocks


def is_rebalance_day(current_idx: int, t_date: str, next_date: str, rebal_freq: Union[str, int]) -> bool:
    """根据频率配置判断是否再平衡"""
    if str(rebal_freq).lower() == "daily":
        return True

    try:
        step = int(rebal_freq)
        return current_idx % step == 0
    except (ValueError, TypeError):
        pass

    curr_dt = pd.to_datetime(t_date)
    next_dt = pd.to_datetime(next_date) if next_date else None

    if str(rebal_freq).lower() == "weekly":
        return next_dt is None or next_dt.isocalendar().week != curr_dt.isocalendar().week

    if str(rebal_freq).lower() == "monthly":
        return next_dt is None or next_dt.month != curr_dt.month

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

    # 读取策略参数配置
    df_cfg = pd.read_excel(config_path)
    col_map = {str(c).strip().lower(): str(c).strip() for c in df_cfg.columns}
    id_col = col_map.get('strategy_id', 'Strategy_ID')

    strat_rows = df_cfg[df_cfg[id_col].astype(str).str.strip().str.upper() == strategy_id.strip().upper()]
    if strat_rows.empty:
        raise ValueError(f"❌ 在 {config_path} 中未找到策略 [{strategy_id}]！可选ID: {df_cfg[id_col].tolist()}")

    row = strat_rows.iloc[0]

    def get_cfg_val(k, def_val):
        real_k = col_map.get(k.lower())
        if not real_k:
            return def_val
        v = row[real_k]
        return def_val if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v

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

    print("\n" + "=" * 75)
    print(f"🎯【回测配置: 动态跟随期货市值对冲 (多期货 + 融券空股票)】")
    print(f"   策略ID: [{strategy_id}] | 基准: [{benchmark}] | 调仓机制: [{rebal_freq}]")
    print(
        f"   初始规模: {initial_notional / 1e4:.0f} 万元 | 引擎: [{solver_type}] | 超配: {overweight}x | Size差: {delta_sz} | Ind差: {delta_in * 100:.1f}%")
    print("=" * 75)

    trading_days = get_trading_days(CONN_STR, start_date, end_date)
    print(f"📅 A 股有效交易日: {trading_days[0]} -> {trading_days[-1]} (共 {len(trading_days)} 天)")
    if len(trading_days) < 2:
        print("❌ 交易日天数过少，至少需要 2 个交易日！")
        return

    # 全局容器
    current_holdings_qty: Dict[str, float] = {}
    locked_futures_contracts: Optional[int] = None
    locked_futures_multiplier: float = 200.0
    current_futures_symbol: str = "IM"

    current_spot_price: Optional[float] = None

    daily_records: List[Dict] = []
    rebal_log: List[Dict] = []

    prices_cache: Dict[str, Dict[str, float]] = {}
    suspension_cache: Dict[str, set] = {}

    for i in range(len(trading_days) - 1):
        t_date = trading_days[i]
        next_date = trading_days[i + 1]
        rebal_flag = is_rebalance_day(i, t_date, next_date, rebal_freq)

        # 确保 t_date 行情已缓存
        if t_date not in prices_cache:
            p_t, b_t, s_t = get_market_eod_data(CONN_STR, t_date, benchmark)
            prices_cache[t_date] = p_t
            suspension_cache[t_date] = s_t

        # -------------------------------------------------------------
        # A. 调仓优化求解 (Rebalance Layer)
        # -------------------------------------------------------------
        if rebal_flag or not current_holdings_qty:
            print(f"🔄 [{i + 1}/{len(trading_days) - 1}] 日期 {t_date}: 触发再平衡优化 (Rebalance)...")
            indices_dict = load_all_indices_from_db(CONN_STR, t_date)
            matched_name = next((k for k in indices_dict if benchmark.upper() in k.upper()), None)

            if matched_name:
                df_basket = indices_dict[matched_name].copy()
                spot_price = get_index_spot_price_from_db(CONN_STR, matched_name, t_date)
                if spot_price and spot_price > 0:
                    current_spot_price = spot_price

                if locked_futures_contracts is None:
                    _, hedge_info = calculate_futures_hedge_and_adjust_notional(
                        matched_name, initial_notional, spot_price, FUTURES_CONFIG
                    )
                    locked_futures_contracts = hedge_info.get("实际锁定手数(手)", 0)
                    locked_futures_multiplier = float(hedge_info.get("合约乘数", 200.0))
                    current_futures_symbol = hedge_info.get("期货品种", "IM")
                    current_target_notional = float(locked_futures_contracts * locked_futures_multiplier * spot_price)
                    print(
                        f"   ⚓ [首日开仓建仓] 锁定期货多头: {locked_futures_contracts} 张 (点位: {spot_price:.2f}) | 目标股票市值: {current_target_notional / 1e4:.2f} 万元")
                else:
                    current_target_notional = float(locked_futures_contracts * locked_futures_multiplier * spot_price)
                    print(
                        f"   🔄 [动态跟随调仓] 维持期货: {locked_futures_contracts} 张 | 点位更新至: {spot_price:.2f} | 动态目标股票市值: {current_target_notional / 1e4:.2f} 万元")

                rebal_log.append({
                    "Rebal_Date": t_date,
                    "Benchmark": matched_name,
                    "Futures_Symbol": current_futures_symbol,
                    "Index_Spot_Price": spot_price,
                    "Locked_Contracts": locked_futures_contracts,
                    "Contract_Multiplier": locked_futures_multiplier,
                    "Single_Contract_Val": round(spot_price * locked_futures_multiplier, 2),
                    "Target_Stock_Notional": round(current_target_notional, 2),
                    "Is_Initial_Open": 1 if i == 0 else 0
                })

                # 券源底仓
                source_dirs = PathConfig.get_source_dirs(t_date)
                try:
                    df_basket = df_basket.merge(load_internal_inventory(source_dirs, t_date, EXCLUDE_SOURCES),
                                                on='stock_code', how='left')
                except Exception:
                    df_basket['internal_avail_qty'] = 0.0
                df_basket['internal_avail_qty'] = pd.to_numeric(
                    df_basket.get('internal_avail_qty', 0.0), errors='coerce'
                ).fillna(0.0)

                # 市值与行业防爆清洗
                df_cap = fetch_stock_market_cap_from_db(CONN_STR, df_basket['stock_code'].tolist(), t_date)
                df_basket = df_basket.merge(df_cap, on='stock_code', how='left')
                df_basket['market_cap'] = pd.to_numeric(df_basket['market_cap'], errors='coerce')
                valid_median_cap = df_basket.loc[df_basket['market_cap'] > 0, 'market_cap'].median()
                valid_median_cap = valid_median_cap if pd.notna(valid_median_cap) and valid_median_cap > 0 else 1e10
                df_basket['market_cap'] = df_basket['market_cap'].fillna(valid_median_cap).apply(
                    lambda x: x if x > 0 else valid_median_cap
                )

                df_basket = df_basket.merge(load_bics_industry_mapping(t_date), on='stock_code', how='left')
                df_basket['bics_level_2'] = df_basket['bics_level_2'].fillna('Others').replace('', 'Others')

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
                    df_model=df_basket, adjusted_notional=current_target_notional, solver_type=solver_type,
                    delta_size=delta_sz, delta_ind=delta_in, overweight_factor=overweight,
                    enable_elastic=True, lambda_ind=lambda_ind, lambda_size=lambda_size
                )

                valid_holds = df_opt[df_opt['opt_qty'] > 0]
                current_holdings_qty = dict(zip(valid_holds['stock_code'], valid_holds['opt_qty']))

                assert len(current_holdings_qty) > 0, "❌ 建仓后持仓为空，请检查优化器输入！"
                assert current_target_notional > 0, "❌ 目标市值不大于0，请检查期货锁仓手数！"

        # -------------------------------------------------------------
        # B. 结算 t -> t+1 日表现与指标留存
        # -------------------------------------------------------------
        if next_date not in prices_cache:
            p_t1, b_t1, s_t1 = get_market_eod_data(CONN_STR, next_date, benchmark)
            prices_cache[next_date] = p_t1
            suspension_cache[next_date] = s_t1

        prices_t = prices_cache[t_date]
        prices_t1 = prices_cache[next_date]
        suspensions_t1 = suspension_cache[next_date]

        _, bmk_spot_t, _ = get_market_eod_data(CONN_STR, t_date, benchmark)
        _, bmk_spot_t1, _ = get_market_eod_data(CONN_STR, next_date, benchmark)

        if bmk_spot_t is None or bmk_spot_t <= 100.0:
            if current_spot_price is None:
                raise ValueError(f"❌ {t_date} 无法获取有效指数点位，且无历史有效值兜底！")
            bmk_spot_t = current_spot_price
            warnings.warn(f"⚠️ {t_date} 指数点位缺失，使用上次有效值 {current_spot_price:.2f}")
        else:
            current_spot_price = bmk_spot_t

        if bmk_spot_t1 is None or bmk_spot_t1 <= 100.0:
            bmk_spot_t1 = bmk_spot_t
            warnings.warn(f"⚠️ {next_date} 指数点位缺失，沿用 {t_date} 的点位 {bmk_spot_t:.2f}")

        short_mv_t = sum(qty * prices_t.get(code, 0.0) for code, qty in current_holdings_qty.items())
        fut_spot_mv_t = locked_futures_contracts * locked_futures_multiplier * bmk_spot_t
        mv_diff_t = fut_spot_mv_t - short_mv_t

        if i == 0:
            daily_records.append({
                "Trade_Date": t_date,
                "Is_Rebal_Day": 1,
                "Index_Spot_Price": round(bmk_spot_t, 2),
                "Index_Return(%)": 0.0,
                "Futures_Symbol": current_futures_symbol,
                "Futures_Multiplier": locked_futures_multiplier,
                "Futures_Contracts": locked_futures_contracts,
                "Futures_Notional_MV": round(fut_spot_mv_t, 2),
                "Futures_PnL": 0.0,
                "Holdings_Count": len(current_holdings_qty),
                "Short_Equity_MV": round(short_mv_t, 2),
                "Short_Equity_PnL": 0.0,
                "Stock_Basket_Return(%)": 0.0,
                "MV_Diff": round(mv_diff_t, 2),
                "Net_Tracking_PnL": 0.0,
                "Net_Tracking_Return(%)": 0.0
            })

        short_equity_pnl = 0.0
        short_mv_t1 = 0.0

        for code, qty in current_holdings_qty.items():
            price_t1 = prices_t1.get(code)
            price_t = prices_t.get(code, 0.0)

            if price_t1 is None or code in suspensions_t1:
                if price_t1 is None and code not in suspensions_t1:
                    warnings.warn(
                        f"⚠️ {next_date} 股票 {code} 无行情且未在停牌表中，按停牌兜底处理（沿用前一日价格 {price_t:.2f}）")
                price_t1 = price_t
            else:
                short_equity_pnl -= qty * (price_t1 - price_t)

            short_mv_t1 += qty * price_t1

        fut_spot_pnl = locked_futures_contracts * locked_futures_multiplier * (bmk_spot_t1 - bmk_spot_t)
        net_tracking_pnl = fut_spot_pnl + short_equity_pnl

        fut_spot_mv_t1 = locked_futures_contracts * locked_futures_multiplier * bmk_spot_t1
        mv_diff_t1 = fut_spot_mv_t1 - short_mv_t1

        net_tracking_ret = (net_tracking_pnl / fut_spot_mv_t) if fut_spot_mv_t > 0 else 0.0
        stock_basket_ret = (-short_equity_pnl / short_mv_t) if short_mv_t > 0 else 0.0
        index_ret = (bmk_spot_t1 / bmk_spot_t) - 1.0

        daily_records.append({
            "Trade_Date": next_date,
            "Is_Rebal_Day": 1 if rebal_flag else 0,
            "Index_Spot_Price": round(bmk_spot_t1, 2),
            "Index_Return(%)": round(index_ret * 100, 4),
            "Futures_Symbol": current_futures_symbol,
            "Futures_Multiplier": locked_futures_multiplier,
            "Futures_Contracts": locked_futures_contracts,
            "Futures_Notional_MV": round(fut_spot_mv_t1, 2),
            "Futures_PnL": round(fut_spot_pnl, 2),
            "Holdings_Count": len(current_holdings_qty),
            "Short_Equity_MV": round(short_mv_t1, 2),
            "Short_Equity_PnL": round(short_equity_pnl, 2),
            "Stock_Basket_Return(%)": round(stock_basket_ret * 100, 4),
            "MV_Diff": round(mv_diff_t1, 2),
            "Net_Tracking_PnL": round(net_tracking_pnl, 2),
            "Net_Tracking_Return(%)": round(net_tracking_ret * 100, 4)
        })

    # =============================================================
    # 3. 统计看板与时序数据增强 (Time Series Augmentation)
    # =============================================================
    df_ts = pd.DataFrame(daily_records)
    print(f"📊 收集完成: 实际生成逐日时序记录共 {len(df_ts)} 行")
    if df_ts.empty:
        print("❌ 未生成任何有效回测结果！")
        return

    df_ts = df_ts.fillna(0.0)

    # 累加与累计净值计算
    df_ts['Index_NAV'] = (1.0 + df_ts['Index_Return(%)'] / 100.0).cumprod()
    df_ts['Stock_Basket_NAV'] = (1.0 + df_ts['Stock_Basket_Return(%)'] / 100.0).cumprod()
    df_ts['Short_NAV'] = (1.0 - df_ts['Stock_Basket_Return(%)'] / 100.0).cumprod()
    df_ts['Hedged_NAV'] = (1.0 + df_ts['Net_Tracking_Return(%)'] / 100.0).cumprod()

    df_ts['Futures_Cum_PnL'] = df_ts['Futures_PnL'].cumsum()
    df_ts['Short_Equity_Cum_PnL'] = df_ts['Short_Equity_PnL'].cumsum()
    df_ts['Net_Tracking_Cum_PnL'] = df_ts['Net_Tracking_PnL'].cumsum()

    # 最大回撤 (MDD) 计算
    df_ts['NAV_Peak'] = df_ts['Hedged_NAV'].cummax()
    df_ts['Drawdown(%)'] = round(((df_ts['Hedged_NAV'] - df_ts['NAV_Peak']) / df_ts['NAV_Peak']) * 100.0, 4)

    # 绩效核心统计
    daily_ret_series = df_ts.iloc[1:]['Net_Tracking_Return(%)'] / 100.0
    tracking_error = daily_ret_series.std() * np.sqrt(252)
    n_days = len(df_ts) - 1
    total_hedged_return = df_ts['Hedged_NAV'].iloc[-1] - 1.0

    annualized_return = (1 + total_hedged_return) ** (252.0 / n_days) - 1 if n_days > 0 else 0.0
    annualized_vol = daily_ret_series.std() * np.sqrt(252)
    info_ratio = (annualized_return / tracking_error) if tracking_error > 0 else 0.0
    max_drawdown = df_ts['Drawdown(%)'].min()

    sharpe_ratio = (annualized_return / annualized_vol) if annualized_vol > 0 else 0.0
    calmar_ratio = (annualized_return / abs(max_drawdown / 100.0)) if max_drawdown != 0 else 0.0

    cum_fut_pnl = df_ts['Futures_PnL'].sum()
    cum_stock_pnl = df_ts['Short_Equity_PnL'].sum()
    cum_net_pnl = df_ts['Net_Tracking_PnL'].sum()
    avg_mv_diff = df_ts['MV_Diff'].abs().mean()
    total_hedged_return_pct = total_hedged_return * 100.0

    # 组装 Performance Summary 看板表
    summary_data = [
        {"指标分类": "1. 基本信息", "指标名称": "策略ID / 挂钩基准", "指标数值": f"{strategy_id} / {benchmark}"},
        {"指标分类": "1. 基本信息", "指标名称": "回测区间 / 调仓机制",
         "指标数值": f"{start_date} ~ {end_date} / {rebal_freq}"},
        {"指标分类": "1. 基本信息", "指标名称": "初始申报本金",
         "指标数值": f"{initial_notional:,.2f} 元 ({initial_notional / 1e4:.0f} 万元)"},
        {"指标分类": "1. 基本信息", "指标名称": "主力期货品种 / 锁定手数",
         "指标数值": f"{current_futures_symbol} ({locked_futures_multiplier:.0f}元/点) | 锁定 {locked_futures_contracts} 张"},
        {"指标分类": "2. 收益与对冲损益", "指标名称": "期货多头端累计盈亏",
         "指标数值": f"{cum_fut_pnl:,.2f} 元 ({cum_fut_pnl / 1e4:+.2f} 万元)"},
        {"指标分类": "2. 收益与对冲损益", "指标名称": "融券空头端累计盈亏",
         "指标数值": f"{cum_stock_pnl:,.2f} 元 ({cum_stock_pnl / 1e4:+.2f} 万元)"},
        {"指标分类": "2. 收益与对冲损益", "指标名称": "期现对冲累计净偏离损益",
         "指标数值": f"{cum_net_pnl:,.2f} 元 ({cum_net_pnl / 1e4:+.2f} 万元)"},
        {"指标分类": "2. 收益与对冲损益", "指标名称": "期现对冲累计净收益率",
         "指标数值": f"{total_hedged_return_pct:+.2f}%"},
        {"指标分类": "2. 收益与对冲损益", "指标名称": "年化对冲超额收益率",
         "指标数值": f"{annualized_return * 100:+.2f}%"},
        {"指标分类": "3. 风险与跟踪评价", "指标名称": "年化跟踪误差 (Tracking Error)",
         "指标数值": f"{tracking_error * 100:.2f}%"},
        {"指标分类": "3. 风险与跟踪评价", "指标名称": "年化信息比率 (Information Ratio)",
         "指标数值": f"{info_ratio:.2f}"},
        {"指标分类": "3. 风险与跟踪评价", "指标名称": "年化夏普比率 (Sharpe Ratio)", "指标数值": f"{sharpe_ratio:.2f}"},
        {"指标分类": "3. 风险与跟踪评价", "指标名称": "卡玛比率 (Calmar Ratio)", "指标数值": f"{calmar_ratio:.2f}"},
        {"指标分类": "3. 风险与跟踪评价", "指标名称": "年化波动率 (Volatility)",
         "指标数值": f"{annualized_vol * 100:.2f}%"},
        {"指标分类": "3. 风险与跟踪评价", "指标名称": "最大回撤 (Max Drawdown)", "指标数值": f"{max_drawdown:.2f}%"},
        {"指标分类": "4. 持仓与偏离风控", "指标名称": "期现名义市值平均偏离",
         "指标数值": f"{avg_mv_diff:,.2f} 元 ({avg_mv_diff / 1e4:.2f} 万元)"},
        {"指标分类": "4. 持仓与偏离风控", "指标名称": "平均融券持仓股票数",
         "指标数值": f"{df_ts['Holdings_Count'].mean():.0f} 支"}
    ]
    df_summary = pd.DataFrame(summary_data)
    df_rebal = pd.DataFrame(rebal_log)
    df_defs = pd.DataFrame(FIELD_DEFINITIONS)

    print("\n" + "=" * 75)
    print(f"🏁【动态跟随期现对冲看板: {strategy_id}】({start_date} -> {end_date} | 调仓: {rebal_freq})")
    print("=" * 75)
    print(f"  🔹 期货持仓端 (指数多头): 累计浮动盈亏 {cum_fut_pnl / 1e4:+.2f} 万元")
    print(f"  🔹 融券持仓端 (股票空头): 累计浮动盈亏 {cum_stock_pnl / 1e4:+.2f} 万元")
    print(f"  🎯 两端对冲净偏离损益:   {cum_net_pnl / 1e4:+.2f} 万元 (累计净偏离率: {total_hedged_return_pct:+.2f}%)")
    print(
        f"  🎯 年化对冲超额: {annualized_return * 100:+.2f}% | 年化 TE: {tracking_error * 100:.2f}% | IR: {info_ratio:.2f} | MDD: {max_drawdown:.2f}%")
    print("=" * 75)

    # =============================================================
    # 4. 导出 Excel 多 Sheet 报表与自适应列宽
    # =============================================================
    out_dir = os.path.join(PROJECT_ROOT, "backtest")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir,
        f"Backtest_{strategy_id}_{benchmark}_{rebal_freq}_{start_date}_{end_date}.xlsx"
    )

    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Performance_Summary', index=False)
        df_ts.to_excel(writer, sheet_name='Daily_TimeSeries', index=False)
        if not df_rebal.empty:
            df_rebal.to_excel(writer, sheet_name='Hedge_Params_Log', index=False)
        df_defs.to_excel(writer, sheet_name='Field_Definitions', index=False)

        # 根据表头（Header）文字长度自适应各列列宽
        for ws in writer.book.worksheets:
            for cell in ws[1]:
                header_text = str(cell.value or '')
                header_len = sum(2 if ord(char) > 127 else 1 for char in header_text)
                col_letter = cell.column_letter
                ws.column_dimensions[col_letter].width = max(header_len + 4, 12)

    print(f"📁 回测报表已生成至:\n   {out_file}\n")

    return df_ts, df_summary, df_rebal, df_defs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy_id")
    parser.add_argument("start_date")
    parser.add_argument("end_date")
    parser.add_argument("rebal_freq", nargs="?", default="daily")
    args = parser.parse_args()
    run_strategy_backtest(
        args.strategy_id, args.start_date, args.end_date, args.rebal_freq
    )