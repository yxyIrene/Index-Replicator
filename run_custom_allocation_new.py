import os
import sys
import re
import pandas as pd
import numpy as np

from config.config import (
    CONN_STR,
    RESTRICTION_SHARE_DIR,
    FUTURES_CONFIG,
    OVERWEIGHT_FACTOR,
    DELTA_SIZE,
    DELTA_IND,
    EXCLUDE_SOURCES
)
from replication.date_utils import PathConfig
from replication.index_loader import load_all_indices, get_index_spot_price_from_db, load_all_indices_from_db
from replication.data_loaders import (
    fetch_stock_market_cap_from_db, fetch_st_stocks_from_db,
    load_jump_restriction_list, load_bics_industry_mapping,
    load_located_list, load_internal_inventory
)
from replication.hedge_calculator import get_index_spot_price, calculate_futures_hedge_and_adjust_notional
from replication.solvers import solve_hedging_portfolio
from replication.reporting import generate_allocation_reports
from replication.locate_demand_generator import generate_dual_locate_demand_lists


def parse_arguments():
    if len(sys.argv) < 4:
        print("用法: python run_custom_allocation_new.py <交易日期> <指数代码> <Notional金额> [LP/QP]")
        sys.exit(1)
    trade_date = re.sub(r'[^0-9]', '', str(sys.argv[1]))
    index_arg = str(sys.argv[2]).strip()
    n_str = str(sys.argv[3]).strip().lower()
    notional = float(re.sub(r'[^\d.]', '', n_str)) * (
        1e8 if '亿' in n_str or 'e8' in n_str else (1e4 if '万' in n_str or 'w' in n_str else 1.0))
    solver_type = str(sys.argv[4]).strip().upper() if len(sys.argv) >= 5 else 'LP'
    return trade_date, index_arg, notional, solver_type


def run_custom_allocation():
    """模式 B: 手工单跑/调试模式"""
    trade_date, index_arg, initial_notional, solver_type = parse_arguments()

    # 1. 指数成分与对冲规整
    #index_file = PathConfig.get_index_file(trade_date)
    # indices_dict = load_all_indices(index_file)
    indices_dict = load_all_indices_from_db(CONN_STR, trade_date)

    matched_name = next((k for k in indices_dict if index_arg.upper() in k.upper()), None)
    if not matched_name:
        raise ValueError(f"❌ 未能匹配到指数 [{index_arg}]")
    df_basket = indices_dict[matched_name].copy()
    df_basket['stock_code'] = df_basket['stock_code'].astype(str).str.zfill(6)

    #spot_price = get_index_spot_price(pd.ExcelFile(index_file), matched_name)
    spot_price = get_index_spot_price_from_db(CONN_STR, matched_name, trade_date)
    adjusted_notional, hedge_info = calculate_futures_hedge_and_adjust_notional(
        matched_name, initial_notional, spot_price, FUTURES_CONFIG
    )

    # 2. 外部数据关联 (底仓、市值、行业、锁券)
    source_dirs = PathConfig.get_source_dirs(trade_date)
    df_basket = df_basket.merge(load_internal_inventory(source_dirs, trade_date, EXCLUDE_SOURCES), on='stock_code',
                                how='left')
    #print(f"🔍 合并后有券股票数: {(df_basket['effective_avail_qty'] > 0).sum()} / {len(df_basket)}")
    df_basket['internal_avail_qty'] = df_basket['internal_avail_qty'].fillna(0.0)

    df_cap = fetch_stock_market_cap_from_db(CONN_STR, df_basket['stock_code'].tolist(), trade_date)
    df_basket = df_basket.merge(df_cap, on='stock_code', how='left')
    df_basket['market_cap'] = df_basket['market_cap'].fillna(df_basket['market_cap'].median())

    df_basket = df_basket.merge(load_bics_industry_mapping(trade_date), on='stock_code', how='left')
    df_basket['bics_level_2'] = df_basket['bics_level_2'].fillna('Others')

    df_basket = df_basket.merge(load_located_list(trade_date, source_dirs), on='stock_code', how='left')
    df_basket['Located Qty'] = pd.to_numeric(df_basket['Located Qty'], errors='coerce').fillna(0.0)
    df_basket['WA borrow cost'] = pd.to_numeric(df_basket['WA borrow cost'], errors='coerce').fillna(0.0)
    df_basket['effective_avail_qty'] = df_basket['internal_avail_qty'] + df_basket['Located Qty']

    # 3. 风险池拦截 (ST + Restriction List)
    exclude_universe = fetch_st_stocks_from_db(CONN_STR, trade_date) | load_jump_restriction_list(RESTRICTION_SHARE_DIR,
                                                                                                  trade_date)
    mask_exc = df_basket['stock_code'].isin(exclude_universe)
    if mask_exc.sum() > 0:
        df_basket.loc[mask_exc, ['internal_avail_qty', 'Located Qty', 'effective_avail_qty']] = 0.0
        print(f"🚫 成功拦截并封锁 {mask_exc.sum()} 支受限标的。")

    # 4. 优化求解
    df_opt, opt_size, bmk_size, ind_dev = solve_hedging_portfolio(
        df_model=df_basket, adjusted_notional=adjusted_notional, solver_type=solver_type,
        delta_size=DELTA_SIZE, delta_ind=DELTA_IND, overweight_factor=OVERWEIGHT_FACTOR
    )

    # 派生字段计算
    lot_size = df_opt['lot_size'] if 'lot_size' in df_opt.columns else 100.0
    df_opt['target_theoretical_qty'] = np.floor(
        (adjusted_notional * df_opt['weight'] / df_opt['price']) / lot_size) * lot_size
    df_opt['internal_avail_mv'] = df_opt['internal_avail_qty'].fillna(0.0) * df_opt['price']
    df_opt['effective_avail_mv'] = df_opt['effective_avail_qty'] * df_opt['price']
    df_opt['shortage_qty'] = np.maximum(0.0, df_opt['target_theoretical_qty'] - df_opt['opt_qty'])
    df_opt['shortage_mv'] = df_opt['shortage_qty'] * df_opt['price']

    # 5. 报表输出与意向需求单生成
    generate_allocation_reports(
        df_opt, matched_name, hedge_info, adjusted_notional, trade_date,
        solver_type, opt_size, bmk_size, ind_dev, DELTA_SIZE, DELTA_IND, OVERWEIGHT_FACTOR
    )
    generate_dual_locate_demand_lists(
        df_opt=df_opt,
        trade_date=trade_date,
        matched_name=matched_name,
        adjusted_notional=adjusted_notional,
        strat_id=None
    )

def run_strategies_from_excel(
    trade_date: str,
    target_strat_id: str = None,
    config_path: str = "config/Strategy.xlsx"
):
    """
    模式 A: 从 config/Strategy.xlsx 读取策略执行
    - target_strat_id 为 None 或 'all' 时: 批量执行所有策略
    - target_strat_id 为具体值 (如 'Strat_1') 时: 仅精准执行该策略
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 找不到配置文件: {config_path}")

    print(f"📖 正在加载策略配置表: {config_path} ...")
    df_cfg = pd.read_excel(config_path)
    col_map = {str(c).strip().lower(): str(c).strip() for c in df_cfg.columns}

    def get_val(row, target_col, default_val=None):
        target_lower = target_col.lower()
        if target_lower in col_map:
            val = row[col_map[target_lower]]
            return default_val if pd.isna(val) else val
        return default_val

    # 🎯 核心过滤：若指定了 target_strat_id，仅保留匹配的那一行
    id_col_name = col_map.get('strategy_id', 'Strategy_ID')
    if target_strat_id and str(target_strat_id).strip().lower() != 'all':
        df_cfg = df_cfg[df_cfg[id_col_name].astype(str).str.strip().str.upper() == target_strat_id.strip().upper()]
        if df_cfg.empty:
            print(f"❌ 在 {config_path} 中未找到策略 ID 为 [{target_strat_id}] 的配置行！")
            return
        print(f"🎯 已命中单策略执行模式: [{target_strat_id}]")

    # 1. 预加载全局共用数据 (TAICHI DB)
    print("\n📦 正在预加载全天基础数据、券源底仓与风控黑名单...")
    indices_dict = load_all_indices_from_db(CONN_STR, trade_date)
    source_dirs = PathConfig.get_source_dirs(trade_date)

    df_internal = load_internal_inventory(source_dirs, trade_date, EXCLUDE_SOURCES)
    df_ind = load_bics_industry_mapping(trade_date)
    df_located = load_located_list(trade_date, source_dirs)

    st_codes_global = fetch_st_stocks_from_db(CONN_STR, trade_date)
    restricted_codes_global = load_jump_restriction_list(RESTRICTION_SHARE_DIR, trade_date)

    summary_records = []

    # 2. 遍历执行策略 (若过滤后就只有指定的 1 个策略)
    for idx, row in df_cfg.iterrows():
        strat_id = str(get_val(row, 'Strategy_ID', f"Strat_{idx + 1}"))
        benchmark = str(get_val(row, 'Benchmark', '')).strip()

        n_raw = str(get_val(row, 'Notional', '0')).strip().lower()
        if '亿' in n_raw or 'e8' in n_raw:
            initial_notional = float(re.sub(r'[^\d.]', '', n_raw)) * 1e8
        elif '万' in n_raw or 'w' in n_raw or 'k' in n_raw or 'm' in n_raw:
            multiplier = 1e6 if 'm' in n_raw else 1e4
            initial_notional = float(re.sub(r'[^\d.]', '', n_raw)) * multiplier
        else:
            initial_notional = float(re.sub(r'[^\d.]', '', n_raw))

        if not benchmark or initial_notional <= 0:
            print(f"⚠️ [跳过] 策略 [{strat_id}] 未填写有效 Benchmark 或 Notional！")
            continue

        solver_type = str(get_val(row, 'Solver_Type', 'LP')).strip().upper()
        overweight = float(get_val(row, 'Overweight_Factor', OVERWEIGHT_FACTOR))
        delta_sz = float(get_val(row, 'Delta_Size', DELTA_SIZE))
        delta_in = float(get_val(row, 'Delta_Ind', DELTA_IND))
        exclude_st_flag = bool(get_val(row, 'Exclude_ST', True))
        exclude_res_flag = bool(get_val(row, 'Exclude_Restriction', True))
        lambda_ind = float(get_val(row, 'Lambda_Ind', 8.0))
        lambda_size = float(get_val(row, 'Lambda_Size', 30.0))

        print("\n" + "=" * 65)
        print(f"🚀 开始执行策略: [{strat_id}] | 基准: [{benchmark}] | 引擎: [{solver_type}]")
        print(f"   规模: {initial_notional / 1e4:.0f} 万元 | 超配: {overweight}x | Size差: {delta_sz} | Ind差: {delta_in * 100:.1f}%")

        # 匹配基准
        matched_name = next((k for k in indices_dict if benchmark.upper() in k.upper()), None)
        if not matched_name:
            print(f"❌ 未找到基准 [{benchmark}] 的成分数据，跳过该策略。")
            continue

        df_basket = indices_dict[matched_name].copy()
        df_basket['stock_code'] = df_basket['stock_code'].astype(str).str.zfill(6)

        spot_price = get_index_spot_price_from_db(CONN_STR, matched_name, trade_date)
        adjusted_notional, hedge_info = calculate_futures_hedge_and_adjust_notional(
            matched_name, initial_notional, spot_price, FUTURES_CONFIG
        )

        # 合并券源、市值、行业
        df_basket = df_basket.merge(df_internal, on='stock_code', how='left')
        df_basket['internal_avail_qty'] = df_basket['internal_avail_qty'].fillna(0.0)

        df_cap = fetch_stock_market_cap_from_db(CONN_STR, df_basket['stock_code'].tolist(), trade_date)
        df_basket = df_basket.merge(df_cap, on='stock_code', how='left')
        df_basket['market_cap'] = df_basket['market_cap'].fillna(df_basket['market_cap'].median())

        df_basket = df_basket.merge(df_ind, on='stock_code', how='left')
        df_basket['bics_level_2'] = df_basket['bics_level_2'].fillna('Others')

        df_basket = df_basket.merge(df_located, on='stock_code', how='left')
        df_basket['Located Qty'] = pd.to_numeric(df_basket['Located Qty'], errors='coerce').fillna(0.0)
        df_basket['WA borrow cost'] = pd.to_numeric(df_basket['WA borrow cost'], errors='coerce').fillna(0.0)
        df_basket['effective_avail_qty'] = df_basket['internal_avail_qty'] + df_basket['Located Qty']

        # 风控排除
        exclude_pool = set()
        if exclude_st_flag:
            exclude_pool |= st_codes_global
        if exclude_res_flag:
            exclude_pool |= restricted_codes_global

        mask_exc = df_basket['stock_code'].isin(exclude_pool)
        if mask_exc.sum() > 0:
            df_basket.loc[mask_exc, ['internal_avail_qty', 'Located Qty', 'effective_avail_qty']] = 0.0

        # 优化求解
        df_opt, opt_size, bmk_size, ind_dev = solve_hedging_portfolio(
            df_model=df_basket,
            adjusted_notional=adjusted_notional,
            solver_type=solver_type,
            delta_size=delta_sz,
            delta_ind=delta_in,
            overweight_factor=overweight,
            enable_elastic=True,
            lambda_ind=lambda_ind,
            lambda_size=lambda_size
        )

        lot_size = df_opt['lot_size'] if 'lot_size' in df_opt.columns else 100.0
        df_opt['target_theoretical_qty'] = np.floor((adjusted_notional * df_opt['weight'] / df_opt['price']) / lot_size) * lot_size
        df_opt['internal_avail_mv'] = df_opt['internal_avail_qty'].fillna(0.0) * df_opt['price']
        df_opt['effective_avail_mv'] = df_opt['effective_avail_qty'] * df_opt['price']
        df_opt['shortage_qty'] = np.maximum(0.0, df_opt['target_theoretical_qty'] - df_opt['opt_qty'])
        df_opt['shortage_mv'] = df_opt['shortage_qty'] * df_opt['price']

        # 导出单策略文件及需求单 (带上 strat_id 避免覆盖)
        generate_allocation_reports(
            df_opt, matched_name, hedge_info, adjusted_notional, trade_date,
            solver_type, opt_size, bmk_size, ind_dev, delta_sz, delta_in, overweight
        )
        generate_dual_locate_demand_lists(
            df_opt=df_opt,
            trade_date=trade_date,
            matched_name=matched_name,
            adjusted_notional=adjusted_notional,
            strat_id=strat_id
        )

        summary_records.append({
            "Strategy_ID": strat_id,
            "Benchmark": benchmark,
            "Solver": solver_type,
            "Hedge_Contracts": hedge_info['实际锁定手数(手)'],
            "Target_Notional": adjusted_notional,
            "Allocated_MV": df_opt['opt_mv'].sum(),
            "Coverage_Ratio(%)": round(df_opt['opt_mv'].sum() / adjusted_notional * 100, 2),
            "Stock_Count": int((df_opt['opt_qty'] > 0).sum()),
            "Size_Deviation(Z)": round(abs(opt_size - bmk_size), 4),
            "Industry_Deviation(%)": round(ind_dev * 100, 2)
        })

    # 3. 输出汇总监控表
    if summary_records:
        df_sum = pd.DataFrame(summary_records)
        tag = target_strat_id if target_strat_id and target_strat_id.lower() != 'all' else 'Batch'
        sum_path = f"./Strategy_{tag}_Summary_{trade_date}.xlsx"
        df_sum.to_excel(sum_path, index=False)
        print("\n" + "=" * 65)
        print(f"🏁【执行完毕】汇总看板已生成: {sum_path}")
        print("=" * 65 + "\n")

if __name__ == "__main__":
    # 模式 1: 仅输入日期 -> 批量跑 Strategy.xlsx 全部策略
    # 用法: python run_custom_allocation_new.py 20260930
    if len(sys.argv) == 2:
        trade_date_arg = re.sub(r'[^0-9]', '', str(sys.argv[1]))
        run_strategies_from_excel(trade_date=trade_date_arg, target_strat_id=None)

    # 模式 2: 输入日期 + 策略 ID -> 仅从 Strategy.xlsx 执行该指定策略
    # 用法: python run_custom_allocation_new.py 20260930 Strat_1
    #       python run_custom_allocation_new.py 20260930 all
    elif len(sys.argv) == 3:
        trade_date_arg = re.sub(r'[^0-9]', '', str(sys.argv[1]))
        target_id_arg = str(sys.argv[2]).strip()
        run_strategies_from_excel(trade_date=trade_date_arg, target_strat_id=target_id_arg)

    # 模式 3: 手动敲 3~4 个参数 -> 脱离 Strategy.xlsx 的临时单跑模式
    # 用法: python run_custom_allocation_new.py 20260930 csi1000 50000000 LP
    elif len(sys.argv) >= 4:
        run_custom_allocation()

    else:
        print("\n" + "=" * 65)
        print("❌ 参数格式错误！支持的三种运行方式:")
        print("👉 1. 按指定策略 ID 执行 (推荐): python run_custom_allocation_new.py 20260930 Strat_1")
        print("👉 2. 批量执行所有策略:          python run_custom_allocation_new.py 20260930")
        print("👉 3. 命令行临时指定参数调试:    python run_custom_allocation_new.py 20260930 csi1000 50000000 LP")
        print("=" * 65 + "\n")