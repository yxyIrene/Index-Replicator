import os
from datetime import datetime, timedelta
import holidays

ONE_DAY = timedelta(days=1)
cn_holidays = holidays.country_holidays("CN")


def get_previous_trading_day(base_dt: datetime = None) -> str:
    """计算 A 股上一个交易日 (YYYYMMDD)"""
    if base_dt is None:
        base_dt = datetime.today()

    cur_dt = base_dt - ONE_DAY
    while cur_dt.weekday() in holidays.WEEKEND or cur_dt in cn_holidays:
        cur_dt -= ONE_DAY
    return cur_dt.strftime("%Y%m%d")


class PathConfig:
    # 1. 估值表邮件附件目录
    BASE_VALUATION = r"\\htisec.local\hk data\Department\PB\PB\Project Management\Hedge_Data\Email_Attachment"

    # 2. BCTMate DailyReport 目录
    BASE_BCTMATE = r"\\htisec.local\HK Data\Department\PB\PB\BCTMate\DailyReport"

    # 3. D1 Scripts SBL Output 目录
    BASE_SBL_OUTPUT = r"\\htisec.local\HK Data\Department\PB\PB\Project Management\D1 Scripts\sbl\output"

    # 4. 指数成分目录
    BASE_INDEX = r"\\htisec.local\hk data\Department\PB\PB\Trade Data\Index Memb"

    @classmethod
    def get_source_dirs(cls, date_str: str) -> list:
        """返回所有需要扫描的数据源目录"""
        dirs = [
            os.path.join(cls.BASE_VALUATION, date_str),
            os.path.join(cls.BASE_BCTMATE, date_str, "DailyReport"),
            cls.BASE_SBL_OUTPUT  # SBL 目录直接存放在 output 下
        ]
        return dirs

    @classmethod
    def get_index_file(cls, date_str: str) -> str:
        folder = cls.BASE_INDEX
        if not os.path.exists(folder):
            folder = "./index_data"

        for ext in [".xlsx", ".xls", ".csv"]:
            candidate = os.path.join(folder, f"Index Data {date_str}{ext}")
            if os.path.exists(candidate):
                return candidate
        return os.path.join(folder, f"Index Data {date_str}.xlsx")