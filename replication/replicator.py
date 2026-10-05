import pandas as pd
import numpy as np


class IndexReplicator:
    def __init__(self, df_pool: pd.DataFrame, df_index: pd.DataFrame, index_name: str):
        self.df_pool = df_pool.copy()
        self.df_index = df_index.copy()
        self.index_name = index_name

        # 映射券池库存
        pool_dict = dict(zip(self.df_pool['股票代码'].astype(str).str.zfill(6), self.df_pool['汇总持仓数量']))
        self.df_index['pool_qty'] = self.df_index['stock_code'].map(pool_dict).fillna(0.0)

    def evaluate_notional(self, notional: float) -> dict:
        """精确评估单一 Notional 下的拟合指标"""
        df = self.df_index.copy()

        # 1. 理论目标市值与目标整手股数 (按 100 股向下取整)
        df['theoretical_target_mv'] = notional * df['weight']
        df['target_qty'] = np.floor((df['theoretical_target_mv'] / df['price']) / df['lot_size']) * df['lot_size']

        # 2. 实际分配股数与实际满足市值
        df['actual_qty'] = np.minimum(df['target_qty'], df['pool_qty'])
        df['actual_mv'] = df['actual_qty'] * df['price']

        # 3. 缺口金额 (针对理论需求的真实缺口)
        total_actual_mv = df['actual_mv'].sum()
        total_shortage_mv = max(0.0, notional - total_actual_mv)

        # 4. 市值覆盖率：严格以目标 notional 为基准
        mv_coverage = (total_actual_mv / notional * 100.0) if notional > 0 else 0.0

        # 5. 只数与受限统计
        covered_count = int((df['actual_qty'] > 0).sum())
        count_coverage = (covered_count / len(df) * 100.0) if len(df) > 0 else 0.0

        shortage_count = int((df['target_qty'] > df['actual_qty']).sum())
        exhausted_count = int(((df['pool_qty'] > 0) & (df['target_qty'] >= df['pool_qty'])).sum())

        # 6. 权重偏离度
        df['actual_weight'] = df['actual_mv'] / total_actual_mv if total_actual_mv > 0 else 0.0
        weight_drift = float(np.abs(df['actual_weight'] - df['weight']).sum() * 50.0)

        return {
            "notional": float(notional),
            "推荐最优Notional(元)": float(notional),
            "实际拟合市值(元)": round(total_actual_mv, 2),
            "券源缺口金额(元)": round(total_shortage_mv, 2),
            "市值覆盖率(%)": round(mv_coverage, 2),
            "只数覆盖率(%)": round(count_coverage, 2),
            "券源短缺受限只数": shortage_count,
            "库存见顶被吃空只数": exhausted_count,
            "权重偏离度(%)": round(weight_drift, 2)
        }

    def scan_notional_ladder(self) -> pd.DataFrame:
        """生成 1000万 ~ 10亿 的阶梯评估表"""
        ladder_points = [
            1e7, 2e7, 3e7, 5e7, 8e7, 1e8, 1.5e8, 2e8, 2.5e8, 3e8,
            4e8, 5e8, 6e8, 7e8, 8e8, 9e8, 1e9
        ]
        records = [self.evaluate_notional(n) for n in ladder_points]
        return pd.DataFrame(records)

    def find_optimal_notional(
            self,
            min_notional: float = 1e7,  # 起始 1000万
            max_notional: float = 1e9,  # 上限 10亿
            step_notional: float = 5e6,  # 扫描步长 500万
            max_allowed_drop: float = 1.0  # 允许覆盖率最大下降 1.0%
    ) -> tuple:
        """
        寻找最优容量：
        以全局真实峰值覆盖率为基准，在 [全局峰值 - 1.0%] 范围内贪心寻找最大 Notional。
        """
        # 1. 密集扫描
        notionals = np.arange(min_notional, max_notional + step_notional, step_notional)
        all_evals = [self.evaluate_notional(n) for n in notionals]

        if not all_evals:
            default_eval = self.evaluate_notional(min_notional)
            return min_notional, default_eval

        # 2. 🎯 获取全局最高真实覆盖率（避免被小规模整手截断误导）
        peak_coverage = max(e["市值覆盖率(%)"] for e in all_evals)

        # 3. 设定阈值（例如全局最高 85.0%，允许最低 84.0%）
        coverage_threshold = peak_coverage - max_allowed_drop

        # 4. 从大到小反向贪心寻找最大 Notional
        optimal_eval = all_evals[0]
        for e in reversed(all_evals):
            if e["市值覆盖率(%)"] >= coverage_threshold:
                optimal_eval = e
                break

        final_notional = optimal_eval.get("notional", min_notional)
        return final_notional, optimal_eval

    def generate_rebalance_basket(self, notional: float) -> pd.DataFrame:
        """生成成分股分配清单"""
        df = self.df_index.copy()
        df['目标股数'] = np.floor((notional * df['weight'] / df['price']) / df['lot_size']) * df['lot_size']
        df['实际分配股数'] = np.minimum(df['目标股数'], df['pool_qty'])
        df['缺口股数'] = df['目标股数'] - df['实际分配股数']

        df['目标市值'] = df['目标股数'] * df['price']
        df['实际满足市值'] = df['实际分配股数'] * df['price']
        df['缺口市值'] = df['缺口股数'] * df['price']

        df.rename(columns={
            'stock_code': '股票代码',
            'stock_name': '股票名称',
            'weight': '指数权重',
            'price': '最新价格',
            'pool_qty': '券池总可用量'
        }, inplace=True)

        return df[
            ['股票代码', '股票名称', '指数权重', '最新价格', '券池总可用量', '目标股数', '实际分配股数', '缺口股数',
             '目标市值', '实际满足市值', '缺口市值']]