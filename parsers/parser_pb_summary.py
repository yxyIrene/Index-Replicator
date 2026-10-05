import os
import re
import pandas as pd
import numpy as np
from typing import List
from models import StockHolding
from .base import BaseParser


class PBSummaryParser(BaseParser):
    name = "PB_Summary解析器"

    def can_handle(self, file_path: str) -> bool:
        """
        判断是否为 PB Summary 报告：
        1. 文件名包含 PB_Summary (忽略大小写)
        2. 或内部包含 '标的明细' Sheet
        """
        base_name = os.path.basename(file_path).lower()
        if "pb_summary" in base_name or "pbsummary" in base_name:
            return True
        try:
            excel_file = pd.ExcelFile(file_path)
            for s in excel_file.sheet_names:
                clean_s = re.sub(r'[\s（）()_-]', '', str(s))
                if "标的明细" in clean_s:
                    return True
        except Exception:
            pass
        return False

    def parse(self, file_path: str) -> List[StockHolding]:
        excel_file = pd.ExcelFile(file_path)

        # 寻找 '标的明细(存续)' Sheet
        target_sheet = None
        for s in excel_file.sheet_names:
            clean_s = re.sub(r'[\s（）()_-]', '', str(s))
            if "标的明细" in clean_s and "存续" in clean_s:
                target_sheet = s
                break
        if not target_sheet:
            for s in excel_file.sheet_names:
                if "标的明细" in s:
                    target_sheet = s
                    break

        if not target_sheet:
            raise ValueError(f"未在文件 {file_path} 中找到包含'标的明细'的Sheet")

        # 表头在第 2 行 (0-indexed 对应 header=1)
        df = pd.read_excel(file_path, sheet_name=target_sheet, header=1)
        df.columns = [str(c).strip() for c in df.columns]

        required_cols = ["业务性质", "标的代码", "客户头寸方向", "分红调整剩余数量"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"PB Summary 中缺少必要列: {col}")

        # 1. 业务性质 == '对冲'
        cond_biz = df["业务性质"].astype(str).str.strip() == "对冲"

        # 2. 客户头寸方向 == '空头'
        cond_pos = df["客户头寸方向"].astype(str).str.strip() == "空头"

        # 3. 标的代码含有 'Equity' 且含有 ('CH' 或 'C2' 或 'C1')
        code_series = df["标的代码"].astype(str).str.strip()
        cond_equity = code_series.str.contains("Equity", case=False, na=False)
        cond_market = code_series.str.contains(r"CH|C2|C1", case=False, na=False)

        # 综合过滤
        df_filtered = df[cond_biz & cond_pos & cond_equity & cond_market].copy()

        # 4. 提取 股票代码 (前6位) 与 持仓数量 (分红调整剩余数量)
        df_filtered["stock_code"] = df_filtered["标的代码"].astype(str).str.strip().str[:6].str.zfill(6)
        df_filtered["quantity"] = pd.to_numeric(df_filtered["分红调整剩余数量"], errors="coerce").fillna(0.0)

        # 5. 提取 标的名称
        name_col = next((c for c in df_filtered.columns if "标的名称" in c or "证券名称" in c), None)
        if name_col:
            df_filtered["stock_name"] = df_filtered[name_col].astype(str).str.strip()
        else:
            df_filtered["stock_name"] = ""

        # 6. 精确提取【市值(交易货币)】列
        # 兼容可能出现的全半角括号
        mv_col = next((c for c in df_filtered.columns if "市值" in c and "交易货币" in c), None)
        if mv_col:
            # 取绝对值以处理空头可能带负号的情况
            df_filtered["market_value"] = pd.to_numeric(df_filtered[mv_col], errors="coerce").abs().fillna(0.0)
        else:
            df_filtered["market_value"] = 0.0

        # 过滤有效正持仓
        df_valid = df_filtered[df_filtered["quantity"] > 0]

        holdings = []
        for _, row in df_valid.iterrows():
            holdings.append(StockHolding(
                stock_code=str(row["stock_code"]).zfill(6),
                stock_name=str(row["stock_name"]) if str(row["stock_name"]) != "nan" else "",
                quantity=float(row["quantity"]),
                market_value=float(row["market_value"])
            ))

        return holdings