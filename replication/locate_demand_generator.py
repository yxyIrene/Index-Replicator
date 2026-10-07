import os
import pandas as pd
import numpy as np


def generate_dual_locate_demand_lists(
        df_opt: pd.DataFrame,
        trade_date: str,
        matched_name: str,
        adjusted_notional: float,
        output_dir: str = "./",
        strat_id: str = None
):
    """
    生成修正后的双口径外部询券清单:
    - 口径 A (全量理想单): 理论缺口 = max(0, Target - Internal)
    - 口径 B (核心保底单): 优化残差且剔除内部可用库存 = min(max(0, Target - Opt), max(0, Target - Internal))
    """
    df = df_opt.copy()

    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    price = df['price'].values
    target_theoretical_qty = df['target_theoretical_qty'].values
    internal_avail_qty = df['internal_avail_qty'].fillna(0.0).values
    opt_qty = df['opt_qty'].fillna(0.0).values

    # =========================================================================
    # 🎯 1. 口径 A：基准完全物理缺口 (Target - Internal)
    # =========================================================================
    raw_shortage_a = np.maximum(0.0, target_theoretical_qty - internal_avail_qty)
    df['口径A_需求股数'] = np.floor(raw_shortage_a / 100.0) * 100.0
    df['口径A_需求市值'] = df['口径A_需求股数'] * price

    df_locate_a = df[df['口径A_需求股数'] > 0].copy()
    df_locate_a.sort_values(by='口径A_需求市值', ascending=False, inplace=True)

    # =========================================================================
    # 🎯 2. 口径 B：修正后的核心保底单 (模型残差 与 真实物理缺口 取小)
    # =========================================================================
    raw_opt_shortage = np.maximum(0.0, target_theoretical_qty - opt_qty)
    # 核心保护：如果内部本身有底仓(只是被优化器因为因子约束压低了)，严禁向外借券
    raw_shortage_b = np.minimum(raw_opt_shortage, raw_shortage_a)
    df['口径B_需求股数'] = np.floor(raw_shortage_b / 100.0) * 100.0
    df['口径B_需求市值'] = df['口径B_需求股数'] * price

    df_locate_b = df[df['口径B_需求股数'] > 0].copy()
    df_locate_b.sort_values(by='口径B_需求市值', ascending=False, inplace=True)

    # =========================================================================
    # 🎯 3. 统计看板
    # =========================================================================
    mv_a = df_locate_a['口径A_需求市值'].sum()
    mv_b = df_locate_b['口径B_需求市值'].sum()

    dashboard = pd.DataFrame([
        {
            "方案类别": "口径 B (模型残差 - 核心保底单)",
            "策略意图": "模型代偿后仍无法解决的真缺口(已剔除模型低配但内部有券的票)，借券阻力最小",
            "询券股票只数": len(df_locate_b),
            "总询券股数": int(df_locate_b['口径B_需求股数'].sum()),
            "总询券金额(元)": round(mv_b, 2),
            "占总Notional比例(%)": round(mv_b / adjusted_notional * 100.0, 2),
            "优先级": "最高 (保底闭环)"
        },
        {
            "方案类别": "口径 A (基准全量 - 纯指数复刻单)",
            "策略意图": "纯物理缺口，不依赖模型代偿，完全对齐指数1:1复制",
            "询券股票只数": len(df_locate_a),
            "总询券股数": int(df_locate_a['口径A_需求股数'].sum()),
            "总询券金额(元)": round(mv_a, 2),
            "占总Notional比例(%)": round(mv_a / adjusted_notional * 100.0, 2),
            "优先级": "次高 (大宗额度充足时使用)"
        }
    ])

    # =========================================================================
    # 🎯 4. 输出标准外发字段
    # =========================================================================
    cols_a = ['stock_code', 'stock_name', 'bics_level_2', 'price', '口径A_需求股数', '口径A_需求市值', 'weight',
              'target_theoretical_qty', 'internal_avail_qty']
    df_out_a = df_locate_a[cols_a].rename(columns={
        'stock_code': '证券代码', 'stock_name': '证券名称', 'bics_level_2': '所属行业', 'price': '参考现价',
        '口径A_需求股数': '需求申报数量(股)', '口径A_需求市值': '需求预估金额(元)', 'weight': '指数权重',
        'target_theoretical_qty': '理论需求股数', 'internal_avail_qty': '内部底仓现有股数'
    })

    cols_b = ['stock_code', 'stock_name', 'bics_level_2', 'price', '口径B_需求股数', '口径B_需求市值', 'weight',
              'target_theoretical_qty', 'opt_qty', 'internal_avail_qty']
    df_out_b = df_locate_b[cols_b].rename(columns={
        'stock_code': '证券代码', 'stock_name': '证券名称', 'bics_level_2': '所属行业', 'price': '参考现价',
        '口径B_需求股数': '需求申报数量(股)', '口径B_需求市值': '需求预估金额(元)', 'weight': '指数权重',
        'target_theoretical_qty': '理论需求股数', 'opt_qty': '模型已分配股数', 'internal_avail_qty': '内部底仓现有股数'
    })

    prefix = f"{strat_id}_" if strat_id else ""
    output_filename = f"./Locate_Demand_List_{prefix}{matched_name}_{trade_date}.xlsx"
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        dashboard.to_excel(writer, sheet_name="询券方案对比总览", index=False)
        df_out_b.to_excel(writer, sheet_name="口径B_核心保底单", index=False)
        df_out_a.to_excel(writer, sheet_name="口径A_全量理想单", index=False)
        for sname in ["口径B_核心保底单", "口径A_全量理想单"]:
            for cell in writer.sheets[sname]['A']:
                cell.number_format = '@'

    print(f"\n📋 外部询券清单已生成: {os.path.abspath(output_filename)}")
    return output_filename