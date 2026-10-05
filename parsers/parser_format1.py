import re
import pandas as pd
from typing import List
from models import StockHolding
from parsers.base import BaseParser


class Format1Parser(BaseParser):
    """
    适用于：锐天、金锝、量派 等标准估值表
    规则：表头在第4行，科目代码以1102开头且长度为14位，取后6位为股票代码，数量取'数量'列
    """
    name = "格式一(锐天/金锝/量派)"

    def can_handle(self, file_path: str) -> bool:
        if not file_path.lower().endswith(('.xlsx', '.xls', '.csv')):
            return False
        try:
            if file_path.lower().endswith('.csv'):
                df_preview = pd.read_csv(file_path, nrows=6, header=None)
            else:
                df_preview = pd.read_excel(file_path, nrows=6, header=None)

            content = df_preview.to_string()
            return "科目代码" in content and "科目名称" in content and "数量" in content
        except Exception:
            return False

    def parse(self, file_path: str) -> List[StockHolding]:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path, header=3, dtype={'科目代码': str})
        else:
            df = pd.read_excel(file_path, header=3, dtype={'科目代码': str})

        df.columns = [str(col).strip() for col in df.columns]

        if "科目代码" not in df.columns or "数量" not in df.columns:
            raise ValueError(f"文件 {file_path} 缺少 '科目代码' 或 '数量' 列")

        holdings: List[StockHolding] = []

        for _, row in df.iterrows():
            code_raw = str(row.get("科目代码", "")).strip()
            name_raw = str(row.get("科目名称", "")).strip()
            qty_raw = row.get("数量", 0)

            # 严格筛选：1102开头 且 长度为14位（如 11021501300761）
            if code_raw.startswith("1102") and len(code_raw) == 14:
                stock_code = code_raw[-6:]

                if re.match(r"^\d{6}$", stock_code):
                    try:
                        quantity = float(str(qty_raw).replace(",", "").strip() or 0)
                    except (ValueError, TypeError):
                        quantity = 0.0

                    market_val = row.get("市值", None)
                    try:
                        market_val = float(str(market_val).replace(",", "").strip()) if pd.notna(market_val) else None
                    except (ValueError, TypeError):
                        market_val = None

                    # 统一使用显式关键字传参
                    holdings.append(StockHolding(
                        stock_code=stock_code,
                        stock_name=name_raw,
                        quantity=quantity,
                        market_value=market_val
                    ))

        return holdings