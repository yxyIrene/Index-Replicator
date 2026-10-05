import pandas as pd
import numpy as np
from typing import List
from models import StockHolding
from .base import BaseParser


class LimanParser(BaseParser):
    name = "黎曼_中债上清所持仓核对解析器"

    def can_handle(self, file_path: str) -> bool:
        """
        判断是否为黎曼持仓核对表：
        特征：表头（第3行）包含 '核对日期'、'产品代码'、'交易属性'、'核对差额'、'证券数量' 等
        """
        try:
            df_preview = pd.read_excel(file_path, header=None, nrows=6)
            content = " ".join(df_preview.astype(str).values.flatten())
            if (
                    "核对日期" in content or "交易属性" in content or "核对差额" in content) and "证券代码" in content and "证券数量" in content:
                return True
        except Exception:
            pass
        return False

    def parse(self, file_path: str) -> List[StockHolding]:
        # 表头在第 3 行 (0-indexed 对应 header=2)
        df = pd.read_excel(file_path, header=2)

        # 找到两组 证券代码、证券名称、证券数量 对应的列
        code_cols = [c for c in df.columns if "证券代码" in str(c)]
        name_cols = [c for c in df.columns if "证券名称" in str(c)]
        qty_cols = [c for c in df.columns if "证券数量" in str(c)]

        if not code_cols or not qty_cols:
            raise ValueError(f"无法在文件 {file_path} 中定位'证券代码'或'证券数量'列")

        # 1. 提取证券代码（优先取第一列，若为空取第二列）
        raw_code = df[code_cols[0]]
        if len(code_cols) > 1:
            raw_code = raw_code.fillna(df[code_cols[1]])

        # 提取 6 位纯数字 A 股股票代码
        clean_code = raw_code.astype(str).str.extract(r'(\d{6})')[0]

        # 2. 提取证券名称
        if name_cols:
            clean_name = df[name_cols[0]]
            if len(name_cols) > 1:
                clean_name = clean_name.fillna(df[name_cols[1]])
        else:
            clean_name = pd.Series([""] * len(df))

        # 3. 提取数量：取两列中较小的那一个 min(数量1, 数量2)
        qty1 = pd.to_numeric(df[qty_cols[0]], errors='coerce').fillna(0.0)
        if len(qty_cols) > 1:
            qty2 = pd.to_numeric(df[qty_cols[1]], errors='coerce').fillna(0.0)
            clean_qty = np.minimum(qty1, qty2)
        else:
            clean_qty = qty1

        # 4. 组装有效持仓 (代码有效且数量 > 0)
        valid_mask = clean_code.notna() & (clean_qty > 0)
        df_valid = pd.DataFrame({
            'code': clean_code[valid_mask],
            'name': clean_name[valid_mask],
            'qty': clean_qty[valid_mask]
        })

        holdings = []
        for _, row in df_valid.iterrows():
            code = str(row['code']).zfill(6)
            name = str(row['name']).strip() if pd.notna(row['name']) else ""
            qty = float(row['qty'])

            holdings.append(StockHolding(
                stock_code=code,
                stock_name=name,
                quantity=qty,
                market_value=0.0
            ))

        return holdings