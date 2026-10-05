import pandas as pd
from sqlalchemy import create_engine, text

# ==============================================================================
# 🎯 数据库连接配置 (PostgreSQL)
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

def explore_database():
    print("=" * 70)
    print(f"🚀 开始连接 PostgreSQL 数据库: [{DB_CONFIG['database']}] @ {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print("=" * 70)

    try:
        engine = create_engine(CONN_STR)
        with engine.connect() as conn:
            # 1. 检查连接与基础信息
            version = conn.execute(text("SELECT version();")).scalar()
            print(f"✅ 连接成功！数据库系统版本:\n   {version}\n")

            # 2. 查询所有用户表信息及估算数据量
            tables_query = text("""
                SELECT 
                    schemaname,
                    relname AS table_name,
                    n_live_tup AS estimated_rows
                FROM pg_stat_user_tables
                ORDER BY n_live_tup DESC;
            """)
            df_tables = pd.read_sql(tables_query, conn)

            if df_tables.empty:
                print("⚠️ 数据库中未找到任何用户表！请核实 Schema 或用户权限。")
                return

            print(f"📋 共扫描到 {len(df_tables)} 张数据表。按数据量排行如下:")
            print("-" * 70)
            print(f"{'Schema':<15} | {'Table Name':<35} | {'预估行数':<12}")
            print("-" * 70)
            for _, r in df_tables.iterrows():
                print(f"{r['schemaname']:<15} | {r['table_name']:<35} | {r['estimated_rows']:<12}")
            print("-" * 70)

            # 3. 重点锁定可能包含持仓/估值/市值的表 (优先探测包含 gzb, pos, hold, stock, balance 的表)
            keywords = ['gzb', 'pos', 'hold', 'stock', 'share', 'balance', 'valuation', 'report']
            candidate_tables = df_tables[
                df_tables['table_name'].str.lower().apply(lambda x: any(k in x for k in keywords))
            ]

            # 如果没有匹配到关键词，默认挑数据量最大的前 3 张表探查
            if candidate_tables.empty:
                inspect_targets = df_tables.head(3)[['schemaname', 'table_name']].to_dict('records')
            else:
                inspect_targets = candidate_tables.head(5)[['schemaname', 'table_name']].to_dict('records')

            print("\n" + "=" * 70)
            print(f"🔍 开始重点探查疑似业务表结构及数据样例 (共 {len(inspect_targets)} 张表)")
            print("=" * 70)

            for target in inspect_targets:
                schema = target['schemaname']
                tname = target['table_name']
                full_table_name = f'"{schema}"."{tname}"'

                print(f"\n📁 表名: 【 {schema}.{tname} 】")

                # (1) 查询表字段及类型
                col_query = text(f"""
                    SELECT 
                        column_name, 
                        data_type, 
                        is_nullable
                    FROM information_schema.columns 
                    WHERE table_schema = :schema AND table_name = :tname
                    ORDER BY ordinal_position;
                """)
                df_cols = pd.read_sql(col_query, conn, params={"schema": schema, "tname": tname})
                col_list = [f"{r['column_name']} ({r['data_type']})" for _, r in df_cols.iterrows()]
                print(f"   🔹 字段列表 ({len(df_cols)} 个):")
                print(f"      {', '.join(col_list[:12])}{' ...' if len(col_list) > 12 else ''}")

                # (2) 查询前 3 行样例数据
                try:
                    df_sample = pd.read_sql(text(f"SELECT * FROM {full_table_name} LIMIT 3;"), conn)
                    print("   🔹 最新样例数据 (前 3 行):")
                    # 设置打印不截断列
                    with pd.option_context('display.max_columns', 15, 'display.width', 1000):
                        print(df_sample)
                except Exception as e:
                    print(f"   ❌ 读取样例数据失败: {e}")

    except Exception as e:
        print(f"\n❌ 连接或执行失败: {e}")
        print("💡 请确认:")
        print("   1. 是否已开启公司的局域网或 VPN (IP 192.168.116.107 是否可 ping 通)；")
        print("   2. 当前 Python 环境是否已安装依赖：pip install psycopg2-binary sqlalchemy")


if __name__ == "__main__":
    explore_database()