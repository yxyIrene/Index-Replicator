# replication/hedge_calculator.py
import re
import numpy as np
import pandas as pd


def get_index_spot_price(excel_file: pd.ExcelFile, target_sheet: str) -> float:
    """从概览页中读取指定指数点位"""
    overview_sheet = next((name for name in excel_file.sheet_names if
                           any(k in name.lower() for k in ['overview', 'list', 'summary', 'index'])),
                          excel_file.sheet_names[0])
    df_overview = pd.read_excel(excel_file, sheet_name=overview_sheet, header=1)
    df_overview.columns = [str(c).strip() for c in df_overview.columns]
    ticker_col = next((c for c in df_overview.columns if 'ticker' in c.lower()), None)
    price_col = next((c for c in df_overview.columns if 'price' in c.lower() or 'value' in c.lower()), None)

    df_clean = df_overview.dropna(subset=[ticker_col, price_col]).copy()
    df_clean['Ticker_Clean'] = df_clean[ticker_col].astype(str).str.upper()
    target_key = target_sheet.upper().replace(" ", "")

    for _, row in df_clean.iterrows():
        t_val = row['Ticker_Clean'].replace(" ", "")
        if target_key in t_val or t_val.startswith(target_key):
            if float(row[price_col]) > 0: return float(row[price_col])

    clean_digits = re.sub(r'\D', '', target_key)
    if clean_digits:
        for _, row in df_clean.iterrows():
            if clean_digits in row['Ticker_Clean']:
                if float(row[price_col]) > 0: return float(row[price_col])
    raise ValueError(f"❌ 未能匹配到指数 [{target_sheet}] 的点位！")


def calculate_futures_hedge_and_adjust_notional(matched_sheet: str, target_notional: float, index_price: float,
                                                futures_config: dict):
    """计算期货张数并规整现货 Notional"""
    cfg = futures_config.get(matched_sheet, {"symbol": "IM", "multiplier": 200, "name": "默认期货"})
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