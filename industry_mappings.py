import os
import pandas as pd
from datetime import datetime, timedelta
import holidays
from sqlalchemy import create_engine, text

# ==============================================================================
# 🎯 1. 交易日与时间逻辑配置
# ==============================================================================
ONE_DAY = timedelta(days=1)
cn_holidays = holidays.country_holidays("CN")


def get_current_trade_date(dt: datetime = None) -> datetime:
    """
    获取当日的有效交易日/工作日 (若当天是周末或法定节假日，向前推至最近的工作日)
    """
    if dt is None:
        dt = datetime.today()

    current_day = dt
    while current_day.weekday() in holidays.WEEKEND or current_day in cn_holidays:
        current_day -= ONE_DAY
    return current_day


# 获取当日交易日字符串 YYYYMMDD
target_date = get_current_trade_date(datetime.today())
TARGET_DATE_STR = target_date.strftime("%Y%m%d")
print(f"📅 当前运行交易日期: [{TARGET_DATE_STR}]")

# ==============================================================================
# 🎯 2. 数据库与输出路径配置
# ==============================================================================
DB_CONFIG = {
    "host": "192.168.116.107",
    "port": 5432,
    "database": "taichi_ops",
    "user": "postgres",
    "password": "123456",
    "timeout": 30
}
CONN_STR = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?connect_timeout={DB_CONFIG['timeout']}"

OUTPUT_DIR = "Industry Mappings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_XLSX = os.path.join(OUTPUT_DIR, f"industry_mapping_{TARGET_DATE_STR}.xlsx")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"industry_mapping_{TARGET_DATE_STR}.csv")


# ==============================================================================
# 🎯 3. 从 TaichiOpsDB 提取股票 (强制严格 6 位纯数字字符补 0)
# ==============================================================================
def fetch_stock_universe():
    print("=" * 65)
    print("📡 正在从 TaichiOpsDB (md_stock_info) 提取沪深两市股票清单...")
    engine = create_engine(CONN_STR)

    # 在 SQL 层面直接使用 LPAD 强制补充 6 位，确保即使是数据库层面也是 6 位字符串
    query = text("""
        SELECT 
            LPAD(TRIM(code::text), 6, '0') AS stock_code,
            exchange
        FROM public.md_stock_info
        WHERE exchange IN ('SSE', 'SZSE')
          AND (is_delisted IS FALSE OR is_delisted IS NULL)
          AND (code ~ '^[036]')
        ORDER BY stock_code;
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    # Python 端二次加固：转字符串并左补 0 至 6 位
    df['stock_code'] = df['stock_code'].astype(str).str.strip().str.zfill(6)
    df['bbg_ticker'] = df['stock_code'] + " CH Equity"

    print(f"✅ 成功提取到 {len(df)} 支 A 股代码。首尾样例: {df['stock_code'].iloc[0]} ~ {df['stock_code'].iloc[-1]}")
    return df


# ==============================================================================
# 🎯 4. 通过 Bloomberg API 请求 BICS 1/2 级行业
# ==============================================================================
def fetch_bics_from_bbg(tickers: list):
    fields = [
        "BICS_LEVEL_1_SECTOR_NAME",
        "BICS_LEVEL_2_INDUSTRY_GROUP_NAME"
    ]

    print(f"🚀 开始通过 Bloomberg API 拉取 BICS 行业数据 (共 {len(tickers)} 支)...")

    try:
        import pdblp
        con = pdblp.BCon(debug=False, port=8194, timeout=30000)
        con.start()

        batch_size = 500
        results = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i: i + batch_size]
            sub_df = con.ref(batch, fields)
            results.append(sub_df)
            print(f"   已抓取: {min(i + batch_size, len(tickers))} / {len(tickers)}")

        con.stop()
        df_long = pd.concat(results, ignore_index=True)
        df_bbg = df_long.pivot(index='ticker', columns='field', values='value').reset_index()

    except ImportError:
        print("ℹ️ 未安装 pdblp，调用原生 blpapi 接口...")
        import blpapi

        session = blpapi.Session()
        if not session.start():
            raise ConnectionError("无法启动 Bloomberg Session，请确认 Bloomberg Terminal 处于登录状态！")
        if not session.openService("//blp/refdata"):
            raise ConnectionError("无法打开 //blp/refdata 服务")

        refDataService = session.getService("//blp/refdata")
        batch_size = 500
        rows = []

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i: i + batch_size]
            request = refDataService.createRequest("ReferenceDataRequest")
            for t in batch:
                request.append("securities", t)
            for f in fields:
                request.append("fields", f)

            session.sendRequest(request)

            while True:
                ev = session.nextEvent(500)
                if ev.eventType() in [blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE]:
                    for msg in ev:
                        secDataArray = msg.getElement("securityData")
                        for j in range(secDataArray.numValues()):
                            secData = secDataArray.getValueAsElement(j)
                            ticker_val = secData.getElementAsString("security")
                            fieldData = secData.getElement("fieldData")

                            row = {"ticker": ticker_val}
                            for f in fields:
                                row[f] = fieldData.getElementAsString(f) if fieldData.hasElement(f) else None
                            rows.append(row)

                if ev.eventType() == blpapi.Event.RESPONSE:
                    break
            print(f"   已抓取: {min(i + batch_size, len(tickers))} / {len(tickers)}")

        session.stop()
        df_bbg = pd.DataFrame(rows)

    return df_bbg


# ==============================================================================
# 🎯 5. 清洗、导出带日期后缀的文件 (严格保持文本 6 位代码)
# ==============================================================================
def main():
    # 1. 获取全量在市 A 股
    df_base = fetch_stock_universe()

    # 2. 从彭博抓取 BICS 行业
    df_bbg = fetch_bics_from_bbg(df_base['bbg_ticker'].tolist())

    # 3. 合并
    df = df_base.merge(df_bbg, left_on='bbg_ticker', right_on='ticker', how='left')

    # 4. 空白值统一填 Others
    df['bics_level_1'] = df['BICS_LEVEL_1_SECTOR_NAME'].fillna('').astype(str).str.strip()
    df['bics_level_1'] = df['bics_level_1'].replace(['', 'nan', 'None'], 'Others')

    df['bics_level_2'] = df['BICS_LEVEL_2_INDUSTRY_GROUP_NAME'].fillna('').astype(str).str.strip()
    df['bics_level_2'] = df['bics_level_2'].replace(['', 'nan', 'None'], 'Others')

    # 5. 再次确保代码是 6 位文本
    df['stock_code'] = df['stock_code'].astype(str).str.strip().str.zfill(6)

    final_cols = ['stock_code', 'exchange', 'bics_level_1', 'bics_level_2']
    df_final = df[final_cols].copy()

    # 6. 保存为 CSV (强制引文格式，确保用 Excel 打开时不吞 0)
    #df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # 7. 保存为 Excel，使用 openpyxl 明确指定该列为纯文本格式
    with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
        df_final.to_excel(writer, index=False, sheet_name='IndustryMapping')
        worksheet = writer.sheets['IndustryMapping']
        # 遍历第一列 (A 列: stock_code)，显式设置单元格数字格式为 '@' (文本)
        for cell in worksheet['A']:
            cell.number_format = '@'

    print("\n" + "=" * 65)
    print(f"🎉 行业映射表导出成功 (日期: {TARGET_DATE_STR})！")
    print(f"   📂 目标目录: {os.path.abspath(OUTPUT_DIR)}")
    print(f"   📄 Excel:   {os.path.abspath(OUTPUT_XLSX)}")
    print(f"   📄 CSV:     {os.path.abspath(OUTPUT_CSV)}")
    print(f"   📋 字段:    {final_cols}")
    print(f"   📊 总行数:  {len(df_final)} 行")
    print("=" * 65)

    print("\n前 5 行样例 (核对代码前置 0):")
    print(df_final.head(5))


if __name__ == "__main__":
    main()