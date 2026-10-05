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
# 🎯 核心业务规则配置
# ==============================================================================
# 1. 严格券源扣减优先级 (从左到右依次扣减)
SOURCE_PRIORITY = ['INTERNAL', 'GTHT_PRINCIPAL', 'GTHT', 'GS', 'SGCIB', 'HUATAI', 'SWHY']

# 2. 盘中模式需要排除的券商分类
EXCLUDE_SOURCES = ['GTHT','GS', 'SGCIB', 'HUATAI', 'SWHY']

# 3. 各大指数对应股指期货合约乘数与品种标识
FUTURES_CONFIG = {
    "SHSN300": {"symbol": "IF", "multiplier": 300, "name": "沪深300股指期货"},
    "SSE50": {"symbol": "IH", "multiplier": 300, "name": "上证50股指期货"},
    "SH000905": {"symbol": "IC", "multiplier": 200, "name": "中证500股指期货"},
    "CSI1000": {"symbol": "IM", "multiplier": 200, "name": "中证1000股指期货"},
    "SH932000": {"symbol": "IM(替代)", "multiplier": 200, "name": "中证2000(挂钩IM对冲)"}
}


def parse_arguments():
    """解析命令行参数"""
    if len(sys.argv) < 4:
        print("\n" + "=" * 60)
        print("❌ 参数不足！使用方法：")
        print("   python run_custom_allocation.py <交易日期> <指数代码/名称> <初始Notional金额(元)>")
        print("\n📌 示例：")
        print("   python run_custom_allocation.py 20260915 000905 100000000")
        print("   python run_custom_allocation.py 20260915 CSI1000 50000000")
        print("   python run_custom_allocation.py 20260915 SH932000 50000000")
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
    """从 Index List Overview 概览页中读取指定指数的最新点位 (Value Price)"""
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
    """根据意向 Notional 计算最接近的期货整数手 (张数)，反算对齐后的股票目标市值"""
    cfg = FUTURES_CONFIG.get(matched_sheet, {"symbol": "IM", "multiplier": 200, "name": "默认期货"})
    multiplier = cfg["multiplier"]
    symbol = cfg["symbol"]

    single_contract_val = float(index_price * multiplier)
    exact_contracts = target_notional / single_contract_val if single_contract_val > 0 else 0.0
    hedge_contracts = int(np.round(exact_contracts))
    hedge_contracts = max(1, hedge_contracts)

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
    """根据输入的指数代码或名称匹配对应的成分表"""
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
    """
    严格加载指定名称的 Located List.xlsx（严禁模糊匹配）：
    - 证券代码：文本格式，截取前 6 位字符 (str[:6])
    - 数量列：精确匹配 '合约数量(股)'
    - 利率列：匹配 '利率(%)' 或 '费率(%)'
    - 输出：聚合后的 stock_code, Located Qty, WA borrow cost (%)
    """
    located_file = None
    target_names = ["Located List.xlsx", "Located List.xls", f"Located_List_{trade_date}.xlsx"]

    # 1. 仅按严格文件名查找
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
        print("\n" + "!" * 60)
        print("❌ 未在以下目录找到 'Located List.xlsx':")
        for sp in search_paths:
            print(f"   📂 {os.path.abspath(sp)}")
        print("!" * 60 + "\n")
        return pd.DataFrame(columns=['stock_code', 'Located Qty', 'WA borrow cost'])

    print(f"📥 成功检测到并加载锁券清单: {located_file}")

    df_loc = pd.read_excel(located_file, dtype=str)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]

    # 2. 定位代码、数量与利率列
    code_col = next((c for c in df_loc.columns if c == '证券代码' or '证券代码' in c), None)
    qty_col = next((c for c in df_loc.columns if '合约数量' in c or c == '合约数量(股)'), None)
    rate_col = next((c for c in df_loc.columns if '利率' in c or '费率' in c or 'cost' in c.lower() or 'rate' in c.lower()), None)

    if not code_col or not qty_col:
        print(f"❌ 未能找到指定的列！需包含 '证券代码' 和 '合约数量(股)'。当前列名: {list(df_loc.columns)}")
        return pd.DataFrame(columns=['stock_code', 'Located Qty', 'WA borrow cost'])

    # 3. 数据清洗与数值化
    df_loc['stock_code'] = df_loc[code_col].astype(str).str.strip().str[:6].str.zfill(6)
    df_loc['qty'] = pd.to_numeric(df_loc[qty_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)

    # 清洗利率列（若含百分号 '%' 则剥离，统一转为以百分比数值显示的 float，如 4.5% -> 4.5）
    if rate_col:
        clean_rate = df_loc[rate_col].astype(str).str.replace('%', '').str.replace(',', '').str.strip()
        df_loc['rate'] = pd.to_numeric(clean_rate, errors='coerce').fillna(0.0)
    else:
        print("⚠️ 未在 Located List 中找到利率列，WA borrow cost 默认置为 0.0")
        df_loc['rate'] = 0.0

    df_loc['qty_x_rate'] = df_loc['qty'] * df_loc['rate']

    # 4. 按 6 位代码聚合计算加权平均利率
    grouped = df_loc.dropna(subset=['stock_code']).groupby('stock_code', as_index=False).agg(
        total_qty=('qty', 'sum'),
        sum_qty_rate=('qty_x_rate', 'sum')
    )

    grouped['WA borrow cost'] = np.where(
        grouped['total_qty'] > 0,
        grouped['sum_qty_rate'] / grouped['total_qty'],
        0.0
    )

    df_loc_agg = grouped[['stock_code', 'total_qty', 'WA borrow cost']].rename(columns={'total_qty': 'Located Qty'})

    print(f"✅ 成功清洗锁券记录共 {len(df_loc_agg)} 支股票，前 5 行样例:")
    for _, r in df_loc_agg.head(5).iterrows():
        print(f"   🔹 代码: {r['stock_code']} | 数量: {r['Located Qty']:.0f} | 加权借券费率: {r['WA borrow cost']:.2f}%")

    return df_loc_agg


def allocate_inventory_by_priority(df_index: pd.DataFrame, df_holdings: pd.DataFrame, notional: float):
    """按指定优先级执行券源自动分配"""
    df = df_index.copy()

    df['theoretical_target_mv'] = notional * df['weight']
    df['target_qty'] = np.floor((df['theoretical_target_mv'] / df['price']) / df['lot_size']) * df['lot_size']
    df['target_mv'] = df['target_qty'] * df['price']

    active_priorities = [s for s in SOURCE_PRIORITY if s not in EXCLUDE_SOURCES]

    df_pivot = df_holdings.pivot_table(
        index='stock_code',
        columns='fund_name',
        values='quantity',
        aggfunc='sum'
    ).fillna(0.0)

    allocation_cols = {src: [] for src in active_priorities}
    actual_allocated_qty = []

    for _, row in df.iterrows():
        code = row['stock_code']
        needed_qty = row['target_qty']
        rem_need = needed_qty
        stock_total_alloc = 0.0

        for src in active_priorities:
            avail = df_pivot.loc[code, src] if (code in df_pivot.index and src in df_pivot.columns) else 0.0
            alloc = min(rem_need, avail)
            allocation_cols[src].append(alloc)
            stock_total_alloc += alloc
            rem_need -= alloc

        actual_allocated_qty.append(stock_total_alloc)

    for src in active_priorities:
        df[f'分配股数_{src}'] = allocation_cols[src]
        df[f'分配市值_{src}'] = df[f'分配股数_{src}'] * df['price']

    df['实际满足股数'] = actual_allocated_qty
    df['实际满足市值'] = df['实际满足股数'] * df['price']
    df['缺口股数'] = df['target_qty'] - df['实际满足股数']
    df['缺口市值'] = df['缺口股数'] * df['price']

    source_stats = []
    for src in active_priorities:
        total_avail_mv = df_holdings[df_holdings['fund_name'] == src]['market_value'].sum()
        total_used_mv = df[f'分配市值_{src}'].sum()
        source_stats.append({
            "券源名称": src,
            "总可用市值(元)": round(float(total_avail_mv), 2),
            "本次分配市值(元)": round(float(total_used_mv), 2),
            "剩余可用市值(元)": round(float(total_avail_mv - total_used_mv), 2),
            "库存利用率(%)": round((total_used_mv / total_avail_mv * 100.0), 2) if total_avail_mv > 0 else 0.0
        })

    return df, pd.DataFrame(source_stats), active_priorities


def run_custom_allocation():
    trade_date, index_arg, initial_notional = parse_arguments()

    # 1. 扫描并加载指数成分数据
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

    # 2. 自动计算期货合约整数手并对齐现货股票目标市值
    spot_price = get_index_spot_price(excel_file, matched_name)
    adjusted_notional, hedge_info = calculate_futures_hedge_and_adjust_notional(
        matched_name, initial_notional, spot_price
    )

    print("=" * 65)
    print(f"🚀【定制指数券源精准匹配工具 - 期货自动对齐模式】")
    print(f"📅 交易日期: [{trade_date}] | 标的指数: [{matched_name}] (成分股: {len(df_index)} 支)")
    print(f"📈 指数点位: {hedge_info['指数点位']:.2f} | 挂钩合约: [{hedge_info['期货品种']}] (乘数: {hedge_info['合约乘数']})")
    print(f"🎯 单张价值: {hedge_info['单张合约价值(元)'] / 1e4:.2f} 万元 ➔ 理论张数: {hedge_info['理论合约手数']} 手")
    print(f"⚡ 锁定对冲手数: 【 {hedge_info['实际锁定手数(手)']} 手 】")
    print(f"💰 初始需求 Notional: {initial_notional / 1e4:.2f} 万元")
    print(f"✅ 调整后目标现货市值: {adjusted_notional / 1e4:.2f} 万元 (偏差: {hedge_info['对冲金额偏差(元)'] / 1e4:+.2f} 万元)")
    print(f"🔄 扣减优先级: {' -> '.join([s for s in SOURCE_PRIORITY if s not in EXCLUDE_SOURCES])}")
    print("=" * 65)

    # 3. 扫描读取当天的盘中券源文件
    source_dirs = PathConfig.get_source_dirs(trade_date)
    parser = HTIIntradayInventoryParser(exclude_sources=EXCLUDE_SOURCES)
    all_holdings = []

    for val_dir in source_dirs:
        if not os.path.exists(val_dir):
            continue
        files = [f for f in os.listdir(val_dir) if
                 not f.startswith(('~', '.')) and f.lower().endswith(('.xlsx', '.xls'))]
        for filename in files:
            if "intraday" not in filename.lower() or trade_date not in filename:
                continue
            file_path = os.path.join(val_dir, filename)
            try:
                holdings = parser.parse(file_path, exclude_sources=EXCLUDE_SOURCES)
                all_holdings.extend(holdings)
                print(f"✅ 成功加载盘中券源: {filename} ({len(holdings)} 条有效券源记录)")
            except Exception as e:
                print(f"❌ 读取券源文件失败 {filename}: {e}")

    if not all_holdings:
        print(f"❌ 未找到日期 [{trade_date}] 的有效 HTI_Intraday_Inventory 文件，流程终止。")
        return

    df_holdings = pd.DataFrame([h.__dict__ for h in all_holdings])
    df_holdings['stock_code'] = df_holdings['stock_code'].astype(str).str.zfill(6)

    # 4. 执行按对齐后市值优先扣减分配
    df_basket, df_source_summary, active_priorities = allocate_inventory_by_priority(
        df_index, df_holdings, adjusted_notional
    )

    # 5. 加载 Located List 并计算增补列 (数量与目标理论股数取小)
    df_located = load_located_list(trade_date, candidate_dirs=source_dirs)
    df_basket = df_basket.merge(df_located, on='stock_code', how='left')

    # 缺失值补 0，并与目标理论股数 target_qty 取小
    df_basket['Located Qty'] = df_basket['Located Qty'].fillna(0.0)
    df_basket['Located Qty'] = np.minimum(df_basket['Located Qty'], df_basket['target_qty'])
    df_basket['WA borrow cost'] = df_basket['WA borrow cost'].fillna(0.0)

    # 计算 Located MV 与 Filled Weights
    df_basket['Located MV'] = df_basket['Located Qty'] * df_basket['price']
    df_basket['Filled Weights'] = np.where(
        adjusted_notional > 0,
        df_basket['Located MV'] / adjusted_notional,
        0.0
    )

    # 6. 生成综合统计指标与看板
    actual_total_mv = df_basket['实际满足市值'].sum()
    shortage_total_mv = df_basket['缺口市值'].sum()
    coverage_pct = round((actual_total_mv / adjusted_notional * 100.0), 2)
    shortage_stock_count = int((df_basket['缺口股数'] > 0).sum())
    full_covered_stock_count = int((df_basket['缺口股数'] == 0).sum())

    total_located_mv = df_basket['Located MV'].sum()
    total_filled_weight = df_basket['Filled Weights'].sum()

    # 计算整体组合的加权借券成本 (按有效 Located MV 加权)
    portfolio_wa_borrow_cost = (
        (df_basket['Located MV'] * df_basket['WA borrow cost']).sum() / total_located_mv
        if total_located_mv > 0 else 0.0
    )

    summary_dashboard = pd.DataFrame([{
        "指数代码标识": matched_name,
        "挂钩期货合约": hedge_info["期货品种"],
        "合约乘数": hedge_info["合约乘数"],
        "指数估值点位": hedge_info["指数点位"],
        "单张期货名义价值(元)": hedge_info["单张合约价值(元)"],
        "理论期货手数": hedge_info["理论合约手数"],
        "期货锁定手数(手)": hedge_info["实际锁定手数(手)"],
        "初始申报Notional(元)": hedge_info["目标申报Notional(元)"],
        "调整后对冲目标市值(元)": adjusted_notional,
        "现货实际满足市值(元)": round(actual_total_mv, 2),
        "现货券源缺口市值(元)": round(shortage_total_mv, 2),
        "实操市值覆盖率(%)": coverage_pct,
        "Located锁定总市值(元)": round(total_located_mv, 2),
        "Filled权重合计(%)": round(total_filled_weight * 100.0, 4),
        "组合综合加权借券费率(%)": round(portfolio_wa_borrow_cost, 4),
        "完全满足股票数": full_covered_stock_count,
        "存在缺口股票数": shortage_stock_count,
        "指数总成分数": len(df_basket)
    }])

    # 7. 整理导出明细表字段 (增加 Located Qty, Located MV, Filled Weights, WA borrow cost)
    base_cols = [
        'stock_code', 'stock_name', 'weight', 'price',
        'target_qty', '实际满足股数', '缺口股数',
        'target_mv', '实际满足市值', '缺口市值'
    ]
    alloc_cols = [f'分配股数_{src}' for src in active_priorities] + [f'分配市值_{src}' for src in active_priorities]
    located_cols = ['Located Qty', 'Located MV', 'Filled Weights', 'WA borrow cost']

    df_export = df_basket[base_cols + alloc_cols + located_cols].copy()
    df_export.rename(columns={
        'stock_code': '股票代码',
        'stock_name': '股票名称',
        'weight': '指数权重',
        'price': '最新价格',
        'target_qty': '目标理论股数',
        'target_mv': '目标理论市值'
    }, inplace=True)

    # 8. 保存 Excel 决策报表
    output_filename = f"./Allocation_{matched_name}_{hedge_info['实际锁定手数(手)']}Lot_{int(adjusted_notional / 1e4)}W_{trade_date}.xlsx"
    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
        summary_dashboard.to_excel(writer, sheet_name="对冲与期货匹配看板", index=False)
        df_source_summary.to_excel(writer, sheet_name="各券源库存消耗", index=False)
        df_export.to_excel(writer, sheet_name="成分股分配明细", index=False)

    print("\n" + "=" * 65)
    print("📊【券源匹配与期货锁定结果总览】")
    print(f"  🔹 对应期货: {hedge_info['期货品种']} | 锁定建仓: {hedge_info['实际锁定手数(手)']} 手")
    print(f"  🔹 目标股票市值: {adjusted_notional / 1e4:.2f} 万元 | 实际借入: {actual_total_mv / 1e4:.2f} 万元 (覆盖率 {coverage_pct}%)")
    print(f"  🔹 Located 锁定市值: {total_located_mv / 1e4:.2f} 万元 | Filled 权重达: {total_filled_weight * 100:.2f}%")
    print(f"  🔹 需进入第三层因子拟合缺口: {shortage_total_mv / 1e4:.2f} 万元 (涉及 {shortage_stock_count} 支股票)")
    print("\n📦【各券源出借贡献】:")
    for _, r in df_source_summary.iterrows():
        print(f"  🔸 [{r['券源名称']}]: 借出 {r['本次分配市值(元)'] / 1e4:.1f}万 / 总可用 {r['总可用市值(元)'] / 1e4:.1f}万 (利用率 {r['库存利用率(%)']}%)")
    print(f"\n🎉 专属分配清单已生成: {output_filename}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_custom_allocation()