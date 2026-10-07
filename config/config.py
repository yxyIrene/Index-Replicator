# replication/config.py

# 1. 内部与外部对手方券源优先级
SOURCE_PRIORITY = ['INTERNAL', 'GTHT_PRINCIPAL', 'GTHT', 'GS', 'SGCIB', 'HUATAI', 'SWHY']
EXCLUDE_SOURCES = ['GTHT', 'GS', 'SGCIB', 'HUATAI', 'SWHY']

# 2. 股指期货合约配置
FUTURES_CONFIG = {
    "SHSN300": {"symbol": "IF", "multiplier": 300, "name": "沪深300股指期货"},
    "SSE50": {"symbol": "IH", "multiplier": 300, "name": "上证50股指期货"},
    "SH000905": {"symbol": "IC", "multiplier": 200, "name": "中证500股指期货"},
    "CSI1000": {"symbol": "IM", "multiplier": 200, "name": "中证1000股指期货"},
    "SH932000": {"symbol": "IM(替代)", "multiplier": 200, "name": "中证2000(挂钩IM对冲)"}
}

# 3. 数据库连接配置 (PostgreSQL - taichi_ops)
DB_CONFIG = {
    "host": "192.168.116.107",
    "port": 5432,
    "database": "taichi_ops",
    "user": "postgres",
    "password": "123456",
    "timeout": 30
}
CONN_STR = (
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
    f"connect_timeout={DB_CONFIG['timeout']}"
)

# 4. Jump 限制名单共享盘路径
RESTRICTION_SHARE_DIR = r"\\htisec.local\hk data\Department\PB\PB\Trade Data\Jump Trading\restriction_list"

# 5. 风控与优化默认阈值
OVERWEIGHT_FACTOR = 1.2    # 允许个股最大超配基准的 120%
DELTA_SIZE = 0.02          # 市值因子最大绝对偏离 (Z-score)
DELTA_IND = 0.05           # BICS Level 2 全行业绝对偏离总和上限 (5%)

# 6. 指数成分文件路径
INDEX_SHARE_DIR = r"\\htisec.local\HK Data\Department\PB\PB\Trade Data\Index Memb"