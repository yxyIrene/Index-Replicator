import os
import sys
import re
import pandas as pd
from datetime import datetime
from parsers import get_parser
from parsers.parser_intraday_inventory import HTIIntradayInventoryParser
from replication.date_utils import get_previous_trading_day, PathConfig
from replication.index_loader import load_all_indices, TARGET_SHEETS
from replication.replicator import IndexReplicator

# ==============================================================================
# 🎯 1. 运行模式选择 (RUN_MODE)
# ==============================================================================
# "intraday"      -> 模式 1：盘中 7 大券源专属模式 (HTI_Intraday_Inventory_YYYYMMDD，支持排除指定来源)
# "exclusive_eod" -> 模式 2：日终券源专属模式 (HTI_Inventory_YYYYMMDD，拆分为 internal / external)
# "standard"      -> 模式 3：常规全量估值表模式 (多基金估值表 + PB Summary)
# ==============================================================================
RUN_MODE = "intraday"

# 🎯 盘中模式下需要排除的券源分类（填入标签名即可，留空 [] 则全部读取）
# 可选值: 'INTERNAL', 'GS', 'SGCIB', 'HUATAI', 'GTHT_PRINCIPAL', 'GTHT', 'SWHY'
EXCLUDE_INTRADAY_SOURCES = ["GS", "SWHY","SGCIB","HUATAI", 'GTHT']

# ==============================================================================
# 2. 各模式白名单与报告文件前缀配置
# ==============================================================================
MODE_CONFIG = {
    "intraday": {
        "desc": "盘中 7 大券源专属分析模式",
        "whitelist": [
            {"include": ["HTI_Intraday_Inventory_"], "exclude": ["Copy", "copy"]}
        ],
        "output_prefix": "HTI_Intraday_Replication_Report_"
    },
    "exclusive_eod": {
        "desc": "日终 SBL 券源独占模式",
        "whitelist": [
            {"include": ["HTI_Inventory_"], "exclude": ["Copy", "copy", "Intraday", "intraday"]}
        ],
        "output_prefix": "HTI_Inventory_Replication_Report_"
    },
    "standard": {
        "desc": "常规全量估值表与对冲汇总模式",
        "whitelist": [
            {"include": ["黎曼3号", "中债上清所持仓核对"]},
            {"include": ["锐天正则7号私募证券投资基金_4级科目估值表"]},
            {"include": ["锐天正则11号私募证券投资基金_4级科目估值表"]},
            {"include": ["神州稳健5号_对账单"]},
            {"include": ["PB_Summary_"], "exclude": ["Copy", "copy"]},
            {"include": ["HTI_Inventory_"], "exclude": ["Copy", "copy", "Intraday", "intraday"]}
        ],
        "output_prefix": "Hedge_Replication_Report_"
    }
}


def check_file_date_match(filename: str, target_date: str) -> bool:
    """通用 8 位日期精确校验 (YYYYMMDD)"""
    date_match = re.search(r'(20\d{6})', filename)
    if date_match:
        return date_match.group(1) == target_date
    return True


def is_file_in_whitelist(filename: str, whitelist_rules: list) -> bool:
    """白名单匹配与排除词过滤"""
    if not whitelist_rules:
        return True

    filename_clean = filename.strip().lower()

    for rule in whitelist_rules:
        inc_kws = [k.lower() for k in rule.get("include", [])]
        exc_kws = [k.lower() for k in rule.get("exclude", [])]

        if all(kw in filename_clean for kw in inc_kws):
            if exc_kws and any(kw in filename_clean for kw in exc_kws):
                continue
            return True

    return False


def extract_fund_name(filename: str) -> str:
    name_no_ext = os.path.splitext(filename)[0]
    if "HTI_Intraday_Inventory" in name_no_ext:
        return "HTI_Intraday"
    if "HTI_Inventory" in name_no_ext:
        return "HTI_Inventory"
    if "PB_Summary" in name_no_ext:
        return "PB_Summary"
    if "号" in name_no_ext:
        return name_no_ext[:name_no_ext.index("号") + 1].strip()
    if "私募" in name_no_ext:
        return name_no_ext[:name_no_ext.index("私募")].strip()
    return name_no_ext.strip()


def build_stock_index_mapping(indices_dict: dict) -> dict:
    mapping = {}
    for sheet_code, df_index in indices_dict.items():
        codes = set(df_index['stock_code'].astype(str).str.zfill(6))
        for code in codes:
            if code not in mapping:
                mapping[code] = []
            mapping[code].append(sheet_code)
    return {code: "、".join(sorted(names)) for code, names in mapping.items()}


def run_pipeline(custom_date: str = None):
    # 1. 确定运行日期
    if custom_date:
        trade_date = re.sub(r'[^0-9]', '', str(custom_date))
        print(f"🎯 使用指定日期: {trade_date}")
    else:
        trade_date = get_previous_trading_day(datetime.today())
        print(f"📅 默认使用上一 A 股交易日: {trade_date}")

    config = MODE_CONFIG.get(RUN_MODE, MODE_CONFIG["standard"])
    print("=" * 60)
    print(f"🚀【当前运行模式: {RUN_MODE}】 -> {config['desc']}")
    if RUN_MODE == "intraday" and EXCLUDE_INTRADAY_SOURCES:
        print(f"🚫【盘中排除券源】: {EXCLUDE_INTRADAY_SOURCES}")
    print(f"🔍 目标日期锁定: [{trade_date}]")
    print("=" * 60)

    source_dirs = PathConfig.get_source_dirs(trade_date)
    index_file = PathConfig.get_index_file(trade_date)

    # 2. 批量扫描数据源目录
    all_holdings = []
    matched_count = 0
    skipped_count = 0

    for val_dir in source_dirs:
        if not os.path.exists(val_dir):
            continue

        print(f"\n📂 正在扫描目录: {val_dir}")
        files = [f for f in os.listdir(val_dir) if
                 not f.startswith(('~', '.')) and f.lower().endswith(('.xlsx', '.xls', '.csv'))]

        for filename in files:
            file_path = os.path.join(val_dir, filename)
            if not os.path.isfile(file_path):
                continue

            # (1) 日期校验
            if not check_file_date_match(filename, trade_date):
                skipped_count += 1
                continue

            # (2) 白名单校验
            if not is_file_in_whitelist(filename, config["whitelist"]):
                skipped_count += 1
                continue

            # (3) 匹配解析器
            parser = get_parser(file_path, run_mode=RUN_MODE, exclude_sources=EXCLUDE_INTRADAY_SOURCES)
            default_fund_name = extract_fund_name(filename)

            if not parser:
                skipped_count += 1
                continue

            try:
                # (4) 针对盘中解析器显式传入排除列表
                if RUN_MODE == "intraday" and isinstance(parser, HTIIntradayInventoryParser):
                    holdings = parser.parse(file_path, exclude_sources=EXCLUDE_INTRADAY_SOURCES)
                else:
                    holdings = parser.parse(file_path)

                for h in holdings:
                    h.source_file = filename
                    if not getattr(h, 'fund_name', None) or h.fund_name == "":
                        h.fund_name = default_fund_name
                    h.valuation_date = trade_date

                all_holdings.extend(holdings)
                matched_count += 1
                print(f"✅ [{parser.name}] 成功解析 [{filename}]: 共生成 {len(holdings)} 条明细持仓")
            except Exception as e:
                print(f"❌ 解析失败 {filename}: {e}")

    print(f"\n📁 扫描完毕: 命中解析 {matched_count} 个文件，过滤跳过 {skipped_count} 个非目标/历史文件")

    if not all_holdings:
        print("⚠️ 未提取到任何有效持仓数据，流程结束。")
        return

    # 3. 读取指数成分
    indices_dict = {}
    stock_to_index_map = {}
    if os.path.exists(index_file):
        print(f"\n📈 正在读取指数成分文件: {index_file}")
        try:
            indices_dict = load_all_indices(index_file)
            stock_to_index_map = build_stock_index_mapping(indices_dict)
            print(f"✅ 已加载 {len(indices_dict)} 个指数，共映射 {len(stock_to_index_map)} 支成分股")
        except Exception as e:
            print(f"❌ 读取指数成分出错: {e}")
    else:
        print(f"⚠️ 未找到对应日期的指数文件: {index_file}")

    # 4. 构建明细表与过滤规则
    df_detail = pd.DataFrame([h.__dict__ for h in all_holdings])
    df_detail["stock_code"] = df_detail["stock_code"].astype(str).str.zfill(6)
    df_detail["Index"] = df_detail["stock_code"].map(stock_to_index_map).fillna("")

    df_detail["quantity"] = pd.to_numeric(df_detail["quantity"], errors="coerce").fillna(0.0)
    df_detail = df_detail[df_detail["quantity"] > 0].copy()

    cols_order_detail = [
        "source_file", "fund_name", "stock_code", "stock_name",
        "Index", "quantity", "market_value", "valuation_date"
    ]
    df_detail = df_detail[[c for c in cols_order_detail if c in df_detail.columns]]

    # 5. 构建券池聚合表
    df_pool = df_detail.groupby("stock_code").agg(
        股票名称=("stock_name", "first"),
        Index=("Index", "first"),
        汇总持仓数量=("quantity", "sum"),
        汇总市值=("market_value", "sum")
    ).reset_index().rename(columns={"stock_code": "股票代码"})

    cols_order_pool = ["股票代码", "股票名称", "Index", "汇总持仓数量", "汇总市值"]
    df_pool = df_pool[[c for c in cols_order_pool if c in df_pool.columns]]
    df_pool.sort_values(by="汇总持仓数量", ascending=False, inplace=True)

    # 6. 向上广域阶梯拟合与瓶颈分析
    summary_dashboard = []
    replication_ladders = {}
    best_baskets = {}

    if indices_dict:
        for sheet_name in TARGET_SHEETS:
            if sheet_name not in indices_dict:
                continue
            df_index = indices_dict[sheet_name]
            replicator = IndexReplicator(df_pool=df_pool, df_index=df_index, index_name=sheet_name)

            # (a) 向上广域阶梯评估 (1000万 ~ 10亿)
            df_ladder = replicator.scan_notional_ladder()
            replication_ladders[f"{sheet_name}_阶梯评估"] = df_ladder

            # (b) 寻找性价比最优 Notional
            opt_notional, opt_metrics = replicator.find_optimal_notional()
            summary_dashboard.append({
                "指数代码": sheet_name,
                "指数成分股数": len(df_index),
                "推荐最优Notional(元)": opt_notional,
                "实际满足金额(元)": opt_metrics["实际拟合市值(元)"],
                "向上缺口金额(元)": opt_metrics["券源缺口金额(元)"],
                "最高市值覆盖率(%)": opt_metrics["市值覆盖率(%)"],
                "只数覆盖率(%)": opt_metrics["只数覆盖率(%)"],
                "券源受限只数": opt_metrics["券源短缺受限只数"],
                "库存吃空只数": opt_metrics["库存见顶被吃空只数"],
                "权重偏离度(%)": opt_metrics["权重偏离度(%)"]
            })

            # (c) 生成最优规模下的成分股分配及缺口清单
            df_basket = replicator.generate_rebalance_basket(opt_notional)
            best_baskets[f"{sheet_name}_最优清单"] = df_basket

    # 7. 导出报告 Excel
    df_summary = pd.DataFrame(summary_dashboard)
    output_filename = f"./{config['output_prefix']}{trade_date}.xlsx"

    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
        if not df_summary.empty:
            df_summary.to_excel(writer, sheet_name="对冲决策总览", index=False)

        for sheet_name, df_ladder in replication_ladders.items():
            df_ladder.to_excel(writer, sheet_name=sheet_name, index=False)

        for sheet_name, df_basket in best_baskets.items():
            df_basket.to_excel(writer, sheet_name=sheet_name, index=False)

        df_pool.to_excel(writer, sheet_name="券池汇总聚合", index=False)
        df_detail.to_excel(writer, sheet_name="各基金持仓明细", index=False)

    print("\n" + "=" * 60)
    print("📊【今日对冲决策推荐与向上容量概览】")
    if not df_summary.empty:
        for _, r in df_summary.iterrows():
            print(f"  🔹 [{r['指数代码']}]: 推荐规模 {r['推荐最优Notional(元)'] / 1e4:.0f}万 | "
                  f"满足 {r['实际满足金额(元)'] / 1e4:.0f}万 (覆盖率 {r['最高市值覆盖率(%)']}%) | "
                  f"缺口 {r['向上缺口金额(元)'] / 1e4:.0f}万 | "
                  f"吃空股票 {r['库存吃空只数']} 支")
    print(f"\n🎉 [{config['desc']}] 报告已成功生成: {output_filename}")
    print("=" * 60)


if __name__ == "__main__":
    custom_dt = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(custom_dt)