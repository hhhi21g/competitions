import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import shelve

# ================== 路径配置 ==================
CANDIDATES_PATH = "dataset/candidates.parquet"  # 候选集
TRAIN_LABELS_PATH = "dataset/train_labels.parquet"  # train labels
CO_DB_PATH = "dataset/co_visitation.db"  # co-visitation disk db
OUTPUT_PATH = "dataset/train_features.parquet"  # 输出特征
BATCH_SIZE = 500_000  # 分块写入

# ================== 分块加载 train_labels ==================
print("Flattening train_labels for processing...")
train_labels_file = pq.ParquetFile(TRAIN_LABELS_PATH)

# 将 train_labels 转成 (session, aid, ts) 的扁平结构，按 batch 写入临时 parquet
train_labels_flat_path = "dataset/train_labels_flat.parquet"
writer_flat = None

for rg in tqdm(range(train_labels_file.num_row_groups), desc="Processing row groups"):
    batch = train_labels_file.read_row_group(rg).to_pandas()
    buffer_flat = []
    for row in batch.itertuples():
        session = row.session
        labels = row.labels
        for key in ["clicks", "carts", "orders"]:
            for item in labels[key]:
                buffer_flat.append({"session": session, "aid": item["aid"], "ts": item["ts"], "type": key})
    df_flat = pd.DataFrame(buffer_flat)
    table_flat = pa.Table.from_pandas(df_flat)
    if writer_flat is None:
        writer_flat = pq.ParquetWriter(train_labels_flat_path, table_flat.schema)
    writer_flat.write_table(table_flat)

if writer_flat:
    writer_flat.close()

print("train_labels_flat.parquet 已生成.")

# ================== 构建 session dict ==================
print("Loading train_labels_flat into session dict...")
train_labels_flat_file = pq.ParquetFile(train_labels_flat_path)
session_dict = {}

for rg in tqdm(range(train_labels_flat_file.num_row_groups), desc="Building session dict"):
    batch = train_labels_flat_file.read_row_group(rg).to_pandas()
    for row in batch.itertuples():
        session = row.session
        if session not in session_dict:
            session_dict[session] = []
        session_dict[session].append((row.aid, row.ts))

print(f"Loaded {len(session_dict)} sessions.")

# ================== 处理候选集 ==================
print("Processing candidates and generating features...")
reader = pq.ParquetFile(CANDIDATES_PATH)
writer_feat = None
buffer_feat = []

# 打开 co-visitation 磁盘数据库
co_db = shelve.open(CO_DB_PATH)

for rg in range(reader.num_row_groups):
    chunk = reader.read_row_group(rg).to_pandas()
    for row in tqdm(chunk.itertuples(), total=len(chunk),
                    desc=f"Processing row group {rg + 1}/{reader.num_row_groups}"):
        session = row.session
        cand = row.candidate
        base_score = row.score

        true_items = session_dict.get(session, [])
        if not true_items:
            continue

        label = 1 if cand in [aid for aid, _ in true_items] else 0

        # 时间加权特征
        co_scores = []
        co_scores_weighted = []
        session_len = len(true_items)
        max_ts = max([ts for _, ts in true_items]) if session_len > 0 else 0

        for aid, ts in true_items:
            score = co_db.get(f"{aid}_{cand}", 0)
            co_scores.append(score)
            # 时间衰减权重，越近权重越大 (可调 alpha)
            alpha = 0.001
            weight = np.exp(-alpha * (max_ts - ts))
            co_scores_weighted.append(score * weight)

        co_max = np.max(co_scores) if co_scores else 0
        co_mean = np.mean(co_scores) if co_scores else 0
        co_sum = np.sum(co_scores) if co_scores else 0

        co_max_weighted = np.max(co_scores_weighted) if co_scores_weighted else 0
        co_mean_weighted = np.mean(co_scores_weighted) if co_scores_weighted else 0
        co_sum_weighted = np.sum(co_scores_weighted) if co_scores_weighted else 0

        buffer_feat.append({
            "session": session,
            "candidate": cand,
            "base_score": base_score,
            "co_max": co_max,
            "co_mean": co_mean,
            "co_sum": co_sum,
            "co_max_weighted": co_max_weighted,
            "co_mean_weighted": co_mean_weighted,
            "co_sum_weighted": co_sum_weighted,
            "session_len": session_len,
            "label": label
        })

    # 分块写入 parquet
    if len(buffer_feat) >= BATCH_SIZE:
        df_feat = pd.DataFrame(buffer_feat)
        table_feat = pa.Table.from_pandas(df_feat)
        if writer_feat is None:
            writer_feat = pq.ParquetWriter(OUTPUT_PATH, table_feat.schema)
        writer_feat.write_table(table_feat)
        buffer_feat = []

# 写入剩余数据
if buffer_feat:
    df_feat = pd.DataFrame(buffer_feat)
    table_feat = pa.Table.from_pandas(df_feat)
    if writer_feat is None:
        writer_feat = pq.ParquetWriter(OUTPUT_PATH, table_feat.schema)
    writer_feat.write_table(table_feat)

# 关闭 writer 和 co-visitation db
if writer_feat:
    writer_feat.close()
co_db.close()

print(f"训练特征已保存到 {OUTPUT_PATH}")
