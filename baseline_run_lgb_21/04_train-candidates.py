import json
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd
import glob
import pyarrow.parquet as pq
import pyarrow as pa

train_path = "../dataset/train.jsonl"
co_vis_pkl = "output\\co_visitation.pkl"
output_path = "../output/submission.csv"

#
def count_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


total_lines = count_lines(train_path)

# 嵌套字典: 字典里的值还是一个字典
# 外层字典: co_visitation[a], 商品a
# 内层字典: co_visitation[a][b], 表示商品a和商品b的共现次数
# defaultdict: 自动初始化
co_visitation_clicks = defaultdict(lambda: defaultdict(int))
co_visitation_cart = defaultdict(lambda: defaultdict(int))
co_visitation_order = defaultdict(lambda: defaultdict(int))

# 热门统计
global_popular = Counter()

# 构建商品共现字典
with open(train_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, total=total_lines, desc='构建共现矩阵'):
        session = json.loads(line)
        events = session['events']

        # 更新全局热门
        for e in events:
            global_popular[e['aid']] += 1

        aids = [event['aid'] for event in session['events'] if event['type'] == 'clicks']
        if len(aids) < 2:  # 点击数太少则跳过
            continue

        # 只考虑相邻点击
        for i in range(len(aids) - 1):
            a, t1 = events[i]['aid'], events[i]['type']
            b, t2 = events[i + 1]['aid'], events[i + 1]['type']
            if a == b:
                continue

            # clicks共现
            if t1 == 'clicks' and t2 == 'clicks':
                co_visitation_clicks[a][b] += 1
                co_visitation_clicks[b][a] += 1

            # click -> cart
            if t1 == 'clicks' and t2 == 'carts':
                co_visitation_order[a][b] += 1

            # click -> order
            if t1 == 'clicks' and t2 == 'orders':
                co_visitation_order[a][b] += 1


# 每个商品只保留 top 50 共现商品
def trim_topk(matrix, k=50):
    for a in matrix:
        matrix[a] = dict(sorted(matrix[a].items(), key=lambda x: -x[1])[:k])
    return matrix


co_visitation_clicks = trim_topk(co_visitation_clicks)
co_visitation_cart = trim_topk(co_visitation_cart)
co_visitation_order = trim_topk(co_visitation_order)

# 保存共现矩阵(以二进制写入模式打开文件,适用于pickle序列化)
with open(co_vis_pkl, 'wb') as f:
    pickle.dump({
        "clicks": dict(co_visitation_clicks),
        "cart": dict(co_visitation_cart),
        "order": dict(co_visitation_order)
    }, f)

print(f'共现矩阵已保存到 {co_vis_pkl}')

# 热门top50
global_popular_top = [aid for aid, _ in global_popular.most_common(50)]

# 权重
weights = {"clicks": 1.0, "cart": 1.5, "order": 2.0}

batch_size = 100000  # 每10万条写一次，可根据内存情况调节
candidate_rows = []
part_id = 0
total_batches = (total_lines // batch_size) + 1

with open(train_path, 'r', encoding='utf-8') as f, tqdm(f, total=total_lines, desc='生成候选集') as session_pbar, tqdm(
        total=total_batches, desc='写入批次') as batch_pbar:
    for i, line in enumerate(session_pbar):
        session = json.loads(line)
        sid = session['session']

        clicked = [event['aid'] for event in session['events'] if event['type'] == 'clicks']

        candidates = defaultdict(int)
        for idx, aid in enumerate(clicked[-20:]):
            weight = (idx + 1) / 20
            for b, score in co_visitation_clicks.get(aid, {}).items():
                candidates[b] += weight * weights["clicks"] * score
            for b, score in co_visitation_cart.get(aid, {}).items():
                candidates[b] += weight * weights["cart"] * score
            for b, score in co_visitation_order.get(aid, {}).items():
                candidates[b] += weight * weights["order"] * score

        # 取top50作为候选
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

        # 分批写入
        if (i + 1) % batch_size == 0:
            df = pd.DataFrame(candidate_rows)
            df.to_parquet(f"dataset/candidates_part_{i // batch_size}.parquet", index=False)
            candidate_rows = []  # 清空缓存
            part_id += 1
            batch_pbar.update(1)

# 最后一批写入
if candidate_rows:
    df = pd.DataFrame(candidate_rows)
    df.to_parquet(f"dataset/candidates_part_last.parquet", index=False)
    part_id += 1
    batch_pbar.update(1)

print("候选集已分批保存到 dataset/")

files = sorted(glob.glob("dataset/candidates_part_*.parquet"))

writer = None

with tqdm(total=len(files), desc="合并候选集文件") as merge_pbar:
    for f in files:
        df = pd.read_parquet(f)
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter("dataset/candidates.parquet", table.schema)
        writer.write_table(table)
        merge_pbar.update(1)

if writer:
    writer.close()

print("候选集已成功合并保存到 dataset/candidates.parquet")
