import os
import sys
import re
import pandas as pd
import numpy as np
from datetime import datetime

from parsers.parser_intraday_inventory import HTIIntradayInventoryParser
from replication.date_utils import get_previous_trading_day, PathConfig
from replication.index_loader import load_all_indices, TARGET_SHEETS

# ==============================================================================
# 🎯 业务配置与阶梯测试档位
# ==============================================================================
SOURCE_PRIORITY = ['INTERNAL', 'GTHT_PRINCIPAL', 'GTHT', 'GS', 'SGCIB', 'HUATAI', 'SWHY']
EXCLUDE_SOURCES = ['GTHT', 'GS', 'SGCIB', 'HUATAI', 'SWHY']

# 基础压力测试阶梯 (元)
LADDER_NOTIONALS = [
    10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000,
    60_000_000, 70_000_000, 80_000_000, 90_000_000, 100_000_000,
    150_000_000, 200_000_000, 250_000_000, 300_000_000, 350_000_000,
    400_000_000, 450_000_000, 500_000_000, 600_000_000, 700_000_000,
    800_000_000, 900_000_000, 1_000_000_000
]


def evaluate_single_notional(df_index: pd.DataFrame, notional: float):
    """单档位精确模拟计算"""
    df = df_index.copy()

    # 理论需求股数 (按 100 股向下整手截断)
    raw_qty = (notional * df['weight'] / df['price']) / 100.0
    df['理论需求股数'] = np.floor(raw_qty) * 100.0

    # 不足 1 手统计
    less_than_1_lot = (raw_qty < 1.0) & (df['weight'] > 0)

    # 实际分配 (受限于底仓库存)
    df['实际分配股数'] = np.minimum(df['理论需求股数'], df['internal_avail_qty'])
    df['缺口股数'] = df['理论需求股数'] - df['实际分配股数']

    df['实际满足市值'] = df['实际分配股数'] * df['price']
    df['缺口金额'] = df['缺口股数'] * df['price']

    # 状态判定
    limited_mask = (df['internal_avail_qty'] < df['理论需求股数']) & (df['理论需求股数'] > 0)
    exhausted_mask = (df['internal_avail_qty'] > 0) & (df['internal_avail_qty'] <= df['理论需求股数'])

    actual_mv = df['实际满足市值'].sum()
    shortage_mv = df['缺口金额'].sum()
    cov_mv = actual_mv / notional * 100.0

    traded_stocks = (df['实际分配股数'] > 0).sum()
    total_stocks = len(df)
    cov_count = traded_stocks / total_stocks * 100.0

    # 权重偏离度 = 0.5 * sum |w_real - w_bmk|
    real_weights = df['实际满足市值'] / (actual_mv if actual_mv > 0 else 1.0)
    weight_dev = 0.5 * np.sum(np.abs(real_weights - df['weight'])) * 100.0

    metrics = {
        '目标Notional(元)': int(notional),
        '实际拟合市值(元)': int(round(actual_mv)),
        '券源缺口金额(元)': int(round(shortage_mv)),
        '市值覆盖率(%)': round(cov_mv, 2),
        '有效成交只数/总成分股': f"{traded_stocks}/{total_stocks}",
        '只数覆盖率(%)': round(cov_count, 2),
        '不足1手未开仓只数': int(less_than_1_lot.sum()),
        '券源短缺受限只数': int(limited_mask.sum()),
        '库存见顶被吃空只数': int(exhausted_mask.sum()),
        '权重偏离度(%)': round(weight_dev, 2)
    }
    return metrics, df


def find_optimal_notional(df_index: pd.DataFrame, df_ladder: pd.DataFrame) -> tuple:
    """
    寻找最佳 Notional：
    原则：消除小规模时的碎股不足1手问题，在覆盖率最高或边际拐点处向大 Notional 靠拢
    并在峰值区间进行 200万 步长的精细微调扫描
    """
    # 1. 在初筛阶梯中寻找最高覆盖率（且只数覆盖最高）的区域
    # 过滤掉由于金额太小导致 1手未开仓过多 的档位
    valid_ladder = df_ladder[df_ladder['不足1手未开仓只数'] <= max(1, int(len(df_index) * 0.01))]
    if valid_ladder.empty:
        valid_ladder = df_ladder

    best_rough_row = valid_ladder.sort_values(
        by=['市值覆盖率(%)', '只数覆盖率(%)', '目标Notional(元)'],
        ascending=[False, False, False]
    ).iloc[0]

    rough_best_notional = best_rough_row['目标Notional(元)']

    # 2. 在粗选点前后 ±3000万，以 200万 为步长做二次精细网格扫描
    fine_min = max(10_000_000, rough_best_notional - 30_000_000)
    fine_max = rough_best_notional + 30_000_000
    fine_grid = np.arange(fine_min, fine_max + 2_000_000, 2_000_000)

    fine_results = []
    for n in fine_grid:
        m, _ = evaluate_single_notional(df_index, n)
        fine_results.append(m)
    df_fine = pd.DataFrame(fine_results)

    # 优先选择：偏离度最小且覆盖率最高，同时偏向更大规模的点
    best_fine = df_fine.sort_values(
        by=['市值覆盖率(%)', '目标Notional(元)'],
        ascending=[False, False]
    ).iloc[0]

    opt_notional = best_fine['目标Notional(元)']
    _, df_best_detail = evaluate_single_notional(df_index, opt_notional)

    return best_fine, df_best_detail


def run_hedge_replication_evaluation():
    # 1. 解析日期
    if len(sys.argv) > 1:
        trade_date = re.sub(r'[^0-9]', '', str(sys.argv[1]))
    else:
        trade_date = get_previous_trading_day()
        print(f"ℹ️ 未输入日期，默认取前一交易日: {trade_date}")

    # 2. 读取指数文件与券源底仓
    index_file = PathConfig.get_index_file(trade_date)
    indices_dict = load_all_indices(index_file)

    source_dirs = PathConfig.get_source_dirs(trade_date)
    parser = HTIIntradayInventoryParser(exclude_sources=EXCLUDE_SOURCES)
    all_holdings = []

    for val_dir in source_dirs:
        if not os.path.exists(val_dir): continue
        for fn in os.listdir(val_dir):
            if "intraday" in fn.lower() and trade_date in fn:
                all_holdings.extend(parser.parse(os.path.join(val_dir, fn), exclude_sources=EXCLUDE_SOURCES))

    df_holdings = pd.DataFrame([h.__dict__ for h in all_holdings])
    df_holdings['stock_code'] = df_holdings['stock_code'].astype(str).str.zfill(6)

    # 聚合底仓
    df_internal = df_holdings.groupby('stock_code', as_index=False).agg(
        internal_avail_qty=('quantity', 'sum'),
        total_market_val=('market_value', 'sum')
    )

    print("\n" + "=" * 75)
    print(f"🚀【自动化对冲容量阶梯评估与最优 Notional 报告生成】 | 日期: {trade_date}")
    print("=" * 75)

    overview_records = []
    ladder_sheets = {}
    best_detail_sheets = {}

    for idx_name, df_index in indices_dict.items():
        df = df_index.copy()
        df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
        df = df.merge(df_internal, on='stock_code', how='left')
        df['internal_avail_qty'] = df['internal_avail_qty'].fillna(0.0)

        # 跑全量基础阶梯
        ladder_records = []
        for n in LADDER_NOTIONALS:
            m, _ = evaluate_single_notional(df, n)
            ladder_records.append(m)
        df_ladder = pd.DataFrame(ladder_records)
        ladder_sheets[f"{idx_name}_阶梯评估"] = df_ladder

        # 寻优最佳 Notional (避开碎股向大规模靠拢)
        best_metrics, df_best_detail = find_optimal_notional(df, df_ladder)

        overview_records.append({
            '指数代码': idx_name,
            '指数成分股数': len(df),
            '推荐最优Notional(元)': best_metrics['目标Notional(元)'],
            '实际满足金额(元)': best_metrics['实际拟合市值(元)'],
            '向上缺口金额(元)': best_metrics['券源缺口金额(元)'],
            '最高市值覆盖率(%)': best_metrics['市值覆盖率(%)'],
            '只数覆盖率(%)': best_metrics['只数覆盖率(%)'],
            '券源受限只数': best_metrics['券源短缺受限只数'],
            '库存吃空只数': best_metrics['库存见顶被吃空只数'],
            '权重偏离度(%)': best_metrics['权重偏离度(%)']
        })

        # 整理最优清单输出
        detail_export = df_best_detail[[
            'stock_code', 'stock_name', 'weight', 'price',
            'internal_avail_qty', '理论需求股数', '实际分配股数',
            '缺口股数', '实际满足市值', '缺口金额'
        ]].copy()

        # 打标状态
        detail_export['状态'] = np.where(
            detail_export['缺口股数'] == 0, '完全满足(Full Fill)',
            np.where(detail_export['实际分配股数'] > 0, '向上不足(Partial Fill)', '券源不足1手')
        )
        detail_export.rename(columns={
            'stock_code': '股票代码', 'stock_name': '股票名称', 'weight': '指数理论权重',
            'price': '最新价格', 'internal_avail_qty': '券池可用库存',
            '缺口股数': '缺口股数(向上不足)', '缺口金额': '缺口金额(向上不足)'
        }, inplace=True)

        detail_export.sort_values(by='缺口金额(向上不足)', ascending=False, inplace=True)
        best_detail_sheets[f"{idx_name}_最优清单"] = detail_export

    # 3. 写入 Excel (与原报表完全一致的 Sheet 顺序)
    df_overview = pd.DataFrame(overview_records)
    out_file = f"./Hedge_Replication_Report_{trade_date}.xlsx"

    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        df_overview.to_excel(writer, sheet_name="对冲决策总览", index=False)

        # 写入各指数阶梯评估
        for sname, sdata in ladder_sheets.items():
            sdata.to_excel(writer, sheet_name=sname, index=False)

        # 写入各指数最优清单
        for sname, sdata in best_detail_sheets.items():
            sdata.to_excel(writer, sheet_name=sname, index=False)

        # 写入券池汇总与持仓明细
        df_internal.rename(columns={'internal_avail_qty': '汇总持仓数量', 'total_market_val': '汇总市值'}).to_excel(
            writer, sheet_name="券池汇总聚合", index=False
        )
        if not df_holdings.empty:
            df_holdings.to_excel(writer, sheet_name="各基金持仓明细", index=False)

    print(df_overview.to_string(index=False))
    print("\n" + "=" * 75)
    print(f"🎉 报告导出成功: {os.path.abspath(out_file)}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_hedge_replication_evaluation()