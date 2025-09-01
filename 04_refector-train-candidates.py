import pickle
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd
import glob
import pyarrow.parquet as pq
import pyarrow as pa
import os
import json

train_path = "dataset/train.jsonl"
co_vis_pkl = "dataset_1/co_visitation.pkl"
output_dir = "dataset_1"
os.makedirs(output_dir, exist_ok=True)

# ========== 加载已有的共现矩阵 ==========
with open(co_vis_pkl, "rb") as f:
    co_vis = pickle.load(f)

co_visitation_clicks = co_vis["clicks"]
co_visitation_cart = co_vis["cart"]
co_visitation_order = co_vis["order"]
print(f"✅ 共现矩阵已加载 ({co_vis_pkl})")

# ========== 全局热门 (候选补位用) ==========
global_popular = Counter()
with open(train_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="统计全局热门"):
        session = json.loads(line)
        for e in session['events']:
            global_popular[e['aid']] += 1

global_popular_top = [aid for aid, _ in global_popular.most_common(50)]

# ========== 候选集生成 ==========
weights = {"clicks": 1.0, "cart": 1.5, "order": 2.0}
batch_size = 100000
candidate_rows = []
part_id = 0

def count_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

total_lines = count_lines(train_path)
total_batches = (total_lines // batch_size) + 1

with open(train_path, 'r', encoding='utf-8') as f, tqdm(f, total=total_lines, desc='生成候选集') as session_pbar:
    for i, line in enumerate(session_pbar):
        session = json.loads(line)
        sid = session['session']
        events = session['events']

        clicked = [e['aid'] for e in events if e['type'] == 'clicks']
        if not clicked:
            continue

        candidates = defaultdict(int)
        for idx, aid in enumerate(clicked[-20:]):  # 最近 20 次点击
            decay = (idx + 1) / 20  # 时间衰减
            for b, score in co_visitation_clicks.get(aid, {}).items():
                candidates[b] += decay * weights["clicks"] * score
            for b, score in co_visitation_cart.get(aid, {}).items():
                candidates[b] += decay * weights["cart"] * score
            for b, score in co_visitation_order.get(aid, {}).items():
                candidates[b] += decay * weights["order"] * score

        # 取 top50
        top_items = sorted(candidates.items(), key=lambda x: -x[1])[:50]
        for aid, score in top_items:
            candidate_rows.append({
                "session": sid,
                "candidate": aid,
                "score": score,
                "source": "co_vis"
            })

        # 热门补位
        for aid in global_popular_top:
            if aid not in candidates:
                candidate_rows.append({
                    "session": sid,
                    "candidate": aid,
                    "score": 0,
                    "source": "popular"
                })

        # 批量写入
        if (i + 1) % batch_size == 0:
            df = pd.DataFrame(candidate_rows)
            df.to_parquet(f"{output_dir}/candidates_part_{i // batch_size}.parquet", index=False)
            candidate_rows = []
            part_id += 1

# 最后一批
if candidate_rows:
    df = pd.DataFrame(candidate_rows)
    df.to_parquet(f"{output_dir}/candidates_part_last.parquet", index=False)
    part_id += 1

print("✅ 候选集已分批保存")

# ========== 合并文件 ==========
files = sorted(glob.glob(f"{output_dir}/candidates_part_*.parquet"))
writer = None

with tqdm(total=len(files), desc="合并候选集文件") as merge_pbar:
    for f in files:
        df = pd.read_parquet(f)
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(f"{output_dir}/candidates.parquet", table.schema)
        writer.write_table(table)
        merge_pbar.update(1)

if writer:
    writer.close()

print(f"🎉 候选集已成功合并保存到 {output_dir}/candidates.parquet")
