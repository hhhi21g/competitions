import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import pickle
import json

CANDIDATES_PATH = "dataset/candidates.pkl"
TEST_PATH = "../dataset/train.jsonl"
CO_PKL_PATH = "../dataset/co_visitation.pkl"
OUTPUT_PATH = "../dataset/train_features.parquet"
BATCH_SIZE = 500_000

print("Loading train.jsonl into session dict...")
session_dict = {}
with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading test.jsonl"):
        record = json.loads(line)
        session = record["session"]
        events = record["events"]
        session_dict[session] = [(ev["aid"], ev["ts"]) for ev in events]

print(f"Loaded {len(session_dict)} sessions from test.jsonl.")

print("Loading candidates.pkl ...")
candidates_df = pd.read_pickle(CANDIDATES_PATH)
print(f"Candidates shape: {candidates_df.shape}")

print("Loading co_visitation.pkl ...")
with open(CO_PKL_PATH, "rb") as f:
    co_vis = pickle.load(f)
print("Co-visitation matrix loaded.")

print("Processing candidates and generating features...")
writer_feat = None
buffer_feat = []

for row in tqdm(candidates_df.itertuples(), total=len(candidates_df), desc="Processing candidates"):
    session = row.session
    cand = row.candidate
    base_score = row.score

    true_items = session_dict.get(session, [])
    if not true_items:
        continue

    # 时间加权特征
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
        "candidate": cand,  # 候选商品
        "base_score": base_score,
        "co_max": co_max,  # 所有历史商品与候选商品最大共现分数
        "co_mean": co_mean,  # 平均共现分数
        "co_sum": co_sum,  # 总共现分数
        "co_max_weighted": co_max_weighted,  # 时间衰减函数，计算weight
        "co_mean_weighted": co_mean_weighted,
        "co_sum_weighted": co_sum_weighted,
        "session_len": session_len
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

if writer_feat:
    writer_feat.close()

print(f"测试特征已保存到 {OUTPUT_PATH}")
