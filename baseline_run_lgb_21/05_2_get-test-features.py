import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import pickle
import json
import gc

# ================== 路径配置 ==================
TEST_PATH = "../dataset/test.jsonl"  # 测试集
CANDIDATES_PATH = "../dataset/test_candidates.parquet"  # 候选集
CO_PKL_PATH = "../dataset/co_visitation.pkl"  # 共现矩阵 pkl
TEST_FEAT_PATH = "../dataset/test_features.parquet"  # 输出特征
BATCH_SIZE = 500_000  # 分块写入

# ================== 1. 加载共现矩阵 ==================
print("Loading co-visitation matrix from pkl...")
with open(CO_PKL_PATH, "rb") as f:
    co_vis = pickle.load(f)   # {aid1: {aid2: count}}

# ================== 2. 读取测试集 session_dict ==================
print("Loading test.jsonl into session dict...")
session_dict = {}
with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading test.jsonl"):
        record = json.loads(line)
        session = record["session"]
        events = record["events"]  # list of {aid, ts, type}
        session_dict[session] = [(ev["aid"], ev["ts"]) for ev in events]

print(f"Loaded {len(session_dict)} sessions from test.jsonl.")

# ================== 3. 生成测试特征 ==================
print("Processing candidates and generating test features...")
reader = pq.ParquetFile(CANDIDATES_PATH)
writer_feat = None
buffer_feat = []

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

        co_scores = []
        co_scores_weighted = []
        session_len = len(true_items)
        max_ts = max([ts for _, ts in true_items]) if session_len > 0 else 0

        for aid, ts in true_items:
            score = co_vis.get(aid, {}).get(cand, 0)
            co_scores.append(score)
            # 时间衰减权重
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
            "session_len": session_len
        })

    # 分块写入 parquet
    if len(buffer_feat) >= BATCH_SIZE:
        df_feat = pd.DataFrame(buffer_feat)
        table_feat = pa.Table.from_pandas(df_feat)
        if writer_feat is None:
            writer_feat = pq.ParquetWriter(TEST_FEAT_PATH, table_feat.schema)
        writer_feat.write_table(table_feat)
        buffer_feat = []
        gc.collect()

# 写入剩余数据
if buffer_feat:
    df_feat = pd.DataFrame(buffer_feat)
    table_feat = pa.Table.from_pandas(df_feat)
    if writer_feat is None:
        writer_feat = pq.ParquetWriter(TEST_FEAT_PATH, table_feat.schema)
    writer_feat.write_table(table_feat)

if writer_feat:
    writer_feat.close()

print(f"测试特征已保存到 {TEST_FEAT_PATH}")
