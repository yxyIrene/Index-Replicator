# replication/reporting.py
import numpy as np
import pandas as pd

def generate_allocation_reports(
    df_opt: pd.DataFrame,
    matched_name: str,
    hedge_info: dict,
    adjusted_notional: float,
    trade_date: str,
    solver_type: str,
    opt_size: float,
    bmk_size: float,
    ind_dev: float,
    delta_size: float,
    delta_ind: float,
    overweight_factor: float
) -> str:
    """计算对冲后分析指标并导出双 Sheet Excel 报表"""
    # 1. 补全资金与缺口明细
    df_opt = df_opt.copy()
    df_opt['internal_avail_mv'] = df_opt['internal_avail_qty'].fillna(0.0) * df_opt['price']
    df_opt['effective_avail_mv'] = df_opt['effective_avail_qty'] * df_opt['price']
    lot_size = df_opt['lot_size'] if 'lot_size' in df_opt.columns else 100.0
    df_opt['target_theoretical_qty'] = np.floor((adjusted_notional * df_opt['weight'] / df_opt['price']) / lot_size) * lot_size
    df_opt['Located Qty'] = np.minimum(df_opt['Located Qty'], df_opt['target_theoretical_qty'])
    df_opt['Located MV'] = df_opt['Located Qty'] * df_opt['price']
    df_opt['Filled Weights'] = np.where(adjusted_notional > 0, df_opt['Located MV'] / adjusted_notional, 0.0)
    df_opt['shortage_qty'] = np.maximum(0.0, df_opt['target_theoretical_qty'] - df_opt['opt_qty'])
    df_opt['shortage_mv'] = df_opt['shortage_qty'] * df_opt['price']

    # 2. 统计汇总
    total_opt_mv = float(df_opt['opt_mv'].sum())
    total_opt_weight = float(df_opt['opt_weight'].sum())
    coverage_pct = round(total_opt_mv / adjusted_notional * 100.0, 2) if adjusted_notional > 0 else 0.0
    total_located_mv = float(df_opt['Located MV'].sum())
    total_filled_weight = float(df_opt['Filled Weights'].sum())
    wa_cost = float((df_opt['WA borrow cost'] * df_opt['Located MV']).sum() / total_located_mv) if total_located_mv > 0 else 0.0

    active_share_abs = 0.5 * np.sum(np.abs(df_opt['opt_weight'] - df_opt['weight']))
    w_norm = df_opt['opt_weight'] / total_opt_weight if total_opt_weight > 0 else df_opt['opt_weight']
    active_share_rel = 0.5 * np.sum(np.abs(w_norm - df_opt['weight']))

    # 3. 构造看板 DataFrame
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
        "意向加权借券费率(%)": round(wa_cost, 4),
        "完全配置股票数": int((df_opt['opt_qty'] > 0).sum()),
        "存在代偿缺口股票数": int((df_opt['shortage_qty'] > 0).sum()),
        "指数总成分数": len(df_opt)
    }])

    # 4. 导出明细重命名
    export_cols = [
        'stock_code', 'stock_name', 'bics_level_2', 'price', 'weight',
        'opt_weight', 'opt_qty', 'opt_mv',
        'target_theoretical_qty', 'shortage_qty', 'shortage_mv',
        'internal_avail_qty', 'internal_avail_mv',
        'Located Qty', 'Located MV', 'effective_avail_qty', 'effective_avail_mv',
        'Filled Weights', 'WA borrow cost'
    ]
    df_export = df_opt[export_cols].rename(columns={
        'stock_code': '股票代码', 'stock_name': '股票名称', 'bics_level_2': 'BICS二级行业',
        'price': '最新价格', 'weight': '基准权重', 'opt_weight': '优化拟合权重',
        'opt_qty': '优化分配股数', 'opt_mv': '优化分配市值',
        'target_theoretical_qty': '理论完全复制股数', 'shortage_qty': '代偿缺口股数',
        'shortage_mv': '缺口市值', 'internal_avail_qty': '内部底仓可用股数',
        'internal_avail_mv': '内部底仓可用市值', 'Located Qty': '意向Located股数',
        'Located MV': '意向Located市值', 'effective_avail_qty': '总有效可用股数(内部+已锁)',
        'effective_avail_mv': '总有效可用市值', 'Filled Weights': '意向Filled权重',
        'WA borrow cost': '意向借券费率(%)'
    })

    output_filename = f"./Allocation_{solver_type}_{matched_name}_{hedge_info['实际锁定手数(手)']}Lot_{int(adjusted_notional / 1e4)}W_{trade_date}.xlsx"
    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
        summary_dashboard.to_excel(writer, sheet_name="对冲与多因子风控看板", index=False)
        df_export.to_excel(writer, sheet_name="优化成分股明细", index=False)

    print(f"📊【优化总览 - {solver_type}】实际建仓覆盖率: {coverage_pct}% | 市值偏离: {abs(opt_size - bmk_size):.4f} <= {delta_size} | 行业偏离: {ind_dev * 100:.2f}% <= {delta_ind*100}%")
    print(f"🎉 报表已生成: {output_filename}")
    return output_filename