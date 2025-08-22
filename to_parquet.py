import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

train_path = "dataset\\train.jsonl"   # 输入文件
train_out = "dataset\\train.parquet"  # 训练集输出
valid_out = "dataset\\valid.parquet"  # 验证集输出

train_ratio = 0.9
buffer_size = 50000  # 缓冲区大小
train_writer = None
valid_writer = None

with open(train_path, "r", encoding="utf-8") as f:
    # 先数一下总行数，方便进度条
    total_lines = sum(1 for _ in open(train_path, "r", encoding="utf-8"))
    split_idx = int(total_lines * train_ratio)

    f.seek(0)  # 回到文件开头
    buffer = []
    for i, line in enumerate(tqdm(f, total=total_lines, desc="Processing")):
        buffer.append(json.loads(line))

        if len(buffer) >= buffer_size:
            df = pd.DataFrame(buffer)
            table = pa.Table.from_pandas(df)

            if i < split_idx:  # 写入训练集
                if train_writer is None:
                    train_writer = pq.ParquetWriter(train_out, table.schema)
                train_writer.write_table(table)
            else:  # 写入验证集
                if valid_writer is None:
                    valid_writer = pq.ParquetWriter(valid_out, table.schema)
                valid_writer.write_table(table)

            buffer = []

    # 写入最后剩下的 buffer
    if buffer:
        df = pd.DataFrame(buffer)
        table = pa.Table.from_pandas(df)
        if i < split_idx:
            if train_writer is None:
                train_writer = pq.ParquetWriter(train_out, table.schema)
            train_writer.write_table(table)
        else:
            if valid_writer is None:
                valid_writer = pq.ParquetWriter(valid_out, table.schema)
            valid_writer.write_table(table)

# 关闭 writer
if train_writer:
    train_writer.close()
if valid_writer:
    valid_writer.close()

print("✅ 数据已成功写入 parquet！")
