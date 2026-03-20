#!/usr/bin/env python3
import sqlite3
import os
import sys
import json

import csv
import datetime
from typing import Sequence

# -------------csv_path-----------------
csv_path = "/root/autodl-tmp/dir_tzh/lotus_dataset/write_csv_json/updated.csv"
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, '..', 'log','qa_table.db')

FIXED_COLS  = ["question", "question_rewritten"]
# 1. 取首行字段
with open(csv_path, newline="", encoding="utf-8") as f:
    header = next(csv.reader(f))          # header -> list[str]

# 2. 过滤掉固定列
period_columns = [col.strip() for col in header if col.strip() not in FIXED_COLS]

print(period_columns)
    
  

def create_frequent_qa_database(db_path):
    """
    Create a SQLite database for storing frequent QA pairs with proper schema
    
    Args:
        db_path (str): Path where the database file should be created
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)


    # 2. 建表 SQL ⬇︎ —— 先动态拼出所有时间列，类型统一用 REAL
    period_sql = ",\n    ".join([f"{col} TEXT" for col in period_columns])

    # create_table_sql = f"""
    # CREATE TABLE IF NOT EXISTS metrics (
    #     id INTEGER PRIMARY KEY,
    #     metric_code TEXT NOT NULL UNIQUE,   -- 建议用英文/下划线代号，如 NA_SALES
    #     metric_name TEXT NOT NULL,          -- 中文全称，例如“北美地区销量”
    #     category TEXT NOT NULL,             -- 销量 / 门店 / 员工 / 财务
    #     units TEXT,                         -- 辅助说明：辆、人、美元、%
    #     {period_sql}                        -- 动态插入所有时间段字段
    # );
    # """

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS qa_table (
        id INTEGER PRIMARY KEY,
        question TEXT NOT NULL,
        question_rewritten TEXT NOT NULL,  
        last_updated TIMESTAMP,
        updated_by TEXT,
        is_active BOOLEAN DEFAULT TRUE,
        {period_sql}                        -- Dynamically insert all time period fields
    );
    """

    
    # Connect to the database (creates it if it doesn't exist)
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create the frequent_qa_pairs table
        cursor.execute(create_table_sql)
        
        # Set a busy timeout to wait for locks to clear
        cursor.execute('PRAGMA busy_timeout=5000')
        
        conn.commit()
        print(f"Database created successfully at {db_path}")
        
            
    except sqlite3.Error as e:
        print(f"Error creating database: {e}", file=sys.stderr)
        return False
    finally:
        if conn:
            conn.close()
            
    return True



def load_csv_to_qa_table(db_path: str, csv_path: str) -> None:
    """
    批量把 csv 中的数据写入 qa_table。所有字段都以文本形式存储。
    
    Args:
        db_path: SQLite 数据库文件路径
        csv_path: 待导入的 CSV 文件路径
    """
    # ── 0. 连接数据库 ───────────────────────────────────────────
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # ── 2. 组织列顺序 ──────────────────────────────────────────
    # FIXED_COLS  = ["question", "question_rewritten", "category", "metadata"]
    META_COLS   = ["last_updated", "updated_by", "is_active"]

    ALL_COLS    = FIXED_COLS + list(period_columns) + META_COLS

    placeholders = ",".join(["?"] * len(ALL_COLS))
    insert_sql   = f"INSERT INTO qa_table ({','.join(ALL_COLS)}) VALUES ({placeholders})"

    # ── 3. 读 CSV 并准备批量参数  ──────────────────────────────
    rows = []
    current_time = datetime.datetime.now().isoformat()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, 2):  # 从第 2 行开始算数据
            # 3.1 固定列
            values = [row.get(c, "") or None for c in FIXED_COLS]
            # 3.2 period 列：缺失或空字符串写 NULL
            for col in period_columns:
                val = row.get(col, "")
                values.append(val if val != "" else None)
            # 3.3 维护字段
            values += [current_time, "csv_import", True]
            rows.append(values)

    # ── 4. 批量插入  ──────────────────────────────────────────
    cur.executemany(insert_sql, rows)
    conn.commit()
    conn.close()
    print(f"导入完成：{len(rows)} 条记录写入 qa_table")

# 用法示例
# load_csv_to_qa_table("data/lotus_qa.db", "lotus_qa.csv")

# 打印所有非空时间字段的值
def non_empty_periods_to_dict(row_id, db_path):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # 1️⃣ 只把所有 period 列选出来
    col_list = ",".join(period_columns)
    cur.execute(f"SELECT {col_list} FROM qa_table WHERE id = ?", (row_id,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        print(f"id = {row_id} 不存在")
        return

    # 2️⃣ 过滤非空并打印
    non_empty = {col: val for col, val in zip(period_columns, row) if val not in (None, "", "NULL")}
    data_dict = {}
    if non_empty:
        for col, val in non_empty.items():
            print(f"{col}: {val}")
            data_dict[col] = val
        return data_dict    
    else:
        return data_dict

def get_table_columns(db_path: str, table_name: str) -> list[str]:
    """
    Return a list of column names for `table_name` in the SQLite database at `db_path`.
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # PRAGMA table_info returns one row per column; the 2nd field (index 1) is the name
    cur.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cur.fetchall()]

    conn.close()
    return columns




if __name__ == "__main__":
    create_frequent_qa_database(db_path)
    load_csv_to_qa_table(db_path, csv_path)
    cols = get_table_columns(db_path, "qa_table")
    print(cols)

    
    # print(non_empty_periods_to_dict(5,db_path))
