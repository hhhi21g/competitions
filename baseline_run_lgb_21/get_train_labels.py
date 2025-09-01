import pandas as pd
import json
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

train_file = "../dataset/train.jsonl"

batch_size = 500_000
sessions = []
labels = []
batch_idx = 0

with open(train_file, "r") as f:
    total_lines = sum(1 for _ in f)

output_file = "../dataset/train_labels.parquet"
writer = None

with open(train_file, "r") as f:
    for line_idx, line in enumerate(tqdm(f, total=total_lines, desc="Processing sessions")):
        data = json.loads(line)
        session = data["session"]
        events = data["events"]

        # 按时间戳排序，保证时间顺序
        events = sorted(events, key=lambda x: x['ts'])

        clicks = [{"aid": e["aid"], "ts": e["ts"]} for e in events if e["type"] == "clicks"]
        carts = [{"aid": e["aid"], "ts": e["ts"]} for e in events if e["type"] == "carts"]
        orders = [{"aid": e["aid"], "ts": e["ts"]} for e in events if e["type"] == "orders"]


        # 去重保持顺序（只按 aid 去重，保留第一个时间戳）
        def dedup_keep_ts(lst):
            seen = set()
            res = []
            for item in lst:
                if item["aid"] not in seen:
                    seen.add(item["aid"])
                    res.append(item)
            return res


        clicks = dedup_keep_ts(clicks)
        carts = dedup_keep_ts(carts)
        orders = dedup_keep_ts(orders)

        sessions.append(session)
        labels.append({
            "clicks": clicks,
            "carts": carts,
            "orders": orders
        })

        # 批量写入 parquet
        if len(sessions) >= batch_size:
            df_batch = pd.DataFrame({"session": sessions, "labels": labels})
            table = pa.Table.from_pandas(df_batch)
            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema)
            writer.write_table(table)
            batch_idx += 1
            sessions, labels = [], []

# 写入剩余数据
if sessions:
    df_batch = pd.DataFrame({"session": sessions, "labels": labels})
    table = pa.Table.from_pandas(df_batch)
    if writer is None:
        writer = pq.ParquetWriter(output_file, table.schema)
    writer.write_table(table)

# 关闭 writer
if writer:
    writer.close()

print(f"完整 train_labels.parquet 已生成")

