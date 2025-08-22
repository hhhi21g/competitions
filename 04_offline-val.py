import json
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

train_path = "dataset\\train.jsonl"
test_path = "dataset\\test.jsonl"
co_vis_pkl = "output\\co_visitation.pkl"
output_path = "output\\submission.csv"
train_out = "dataset\\train.parquet"
valid_out = "dataset\\valid.parquet"

train_ratio = 0.9
buffer_size = 50000  # 每次写入多少行

train_writer = None
valid_writer = None
buffer = []

# 计算总行数
with open(train_path, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)
split_idx = int(total_lines * train_ratio)

with open(train_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(tqdm(f, total=total_lines, desc="Writing Parquet")):
        buffer.append(json.loads(line))

        if len(buffer) >= buffer_size:
            table = pa.Table.from_pylist(buffer)  # 保留嵌套 events
            buffer = []

            if i < split_idx:
                if train_writer is None:
                    train_writer = pq.ParquetWriter(train_out, table.schema)
                train_writer.write_table(table)
            else:
                if valid_writer is None:
                    valid_writer = pq.ParquetWriter(valid_out, table.schema)
                valid_writer.write_table(table)

# 写入最后剩余的 buffer
if buffer:
    table = pa.Table.from_pylist(buffer)
    if i < split_idx:
        if train_writer is None:
            train_writer = pq.ParquetWriter(train_out, table.schema)
        train_writer.write_table(table)
    else:
        if valid_writer is None:
            valid_writer = pq.ParquetWriter(valid_out, table.schema)
        valid_writer.write_table(table)

if train_writer:
    train_writer.close()
if valid_writer:
    valid_writer.close()

print("✅ Parquet 文件已写入成功！")


def count_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


total_lines = count_lines(train_path)
total_test = count_lines(test_path)

# 嵌套字典: 字典里的值还是一个字典
# 外层字典: co_visitation[a], 商品a
# 内层字典: co_visitation[a][b], 表示商品a和商品b的共现次数
# defaultdict: 自动初始化
co_visitation_clicks = defaultdict(lambda: defaultdict(int))
co_visitation_cart = defaultdict(lambda: defaultdict(int))
co_visitation_order = defaultdict(lambda: defaultdict(int))

# 热门统计
global_popular = Counter()


def process_session(session):
    events = session['events']
    for e in events:
        global_popular[e['aid']] += 1

    aids = [e['aid'] for e in events if e['type'] == 'clicks']
    if len(aids) < 2:
        return

    for i in range(len(events) - 1):
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
            co_visitation_cart[a][b] += 1

        # click -> order
        if t1 == 'clicks' and t2 == 'orders':
            co_visitation_order[a][b] += 1


# 流式处理 parquet
def iterate_parquet(parquet_file):
    pf = pq.ParquetFile(parquet_file)
    for batch in pf.iter_batches(batch_size=buffer_size):
        table = pa.Table.from_batches([batch])
        for record in table.to_pylist():
            yield record


def count_parquet_rows(parquet_file):
    pf = pq.ParquetFile(parquet_file)
    total = sum(batch.num_rows for batch in pf.iter_batches())
    return total


total_train_rows = count_parquet_rows(train_out)

print("构建共现矩阵中...")
for session in tqdm(iterate_parquet(train_out), total=total_train_rows, desc="Building Co-visitation"):
    process_session(session)


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


# 离线验证
def recommend(session):
    clicked = [e['aid'] for e in session['events'] if e['type'] == 'clicks']
    recent_clicks = clicked[-5:]
    candidates = defaultdict(int)

    for aid in recent_clicks:
        for b, score in co_visitation_clicks.get(aid, {}).items():
            candidates[b] += weights["clicks"] * score
        for b, score in co_visitation_cart.get(aid, {}).items():
            candidates[b] += weights["cart"] * score
        for b, score in co_visitation_order.get(aid, {}).items():
            candidates[b] += weights["order"] * score

    # 取top20
    top_items = [str(aid) for aid, _ in sorted(candidates.items(), key=lambda x: -x[1])[:20]]

    # 热门商品补位
    if len(top_items) < 20:
        need = 20 - len(top_items)
        for aid in global_popular_top:
            if str(aid) not in top_items:
                top_items.append(str(aid))
            if len(top_items) == 20:
                break

    return top_items


# 计算recall@20
def recall_at20(true_items, pred_items):
    if len(true_items) == 0:
        return 0
    return len(set(true_items) & set(pred_items)) / len(true_items)


total_val_rows = count_parquet_rows(valid_out)
recalls = {"clicks": [], "carts": [], "orders": []}

for session in tqdm(iterate_parquet(valid_out), total=total_val_rows, desc="验证中..."):
    preds = recommend(session)

    true_clicks = [str(e['aid']) for e in session['events'] if e['type'] == 'clicks']
    true_carts = [str(e['aid']) for e in session['events'] if e['type'] == 'carts']
    true_orders = [str(e['aid']) for e in session['events'] if e['type'] == 'orders']

    recalls["clicks"].append(recall_at20(true_clicks, preds))
    recalls["carts"].append(recall_at20(true_carts, preds))
    recalls["orders"].append(recall_at20(true_orders, preds))

# 取平均
recall_clicks = sum(recalls["clicks"]) / len(recalls["clicks"])
recall_carts = sum(recalls["carts"]) / len(recalls["carts"])
recall_orders = sum(recalls["orders"]) / len(recalls["orders"])

score = 0.10 * recall_clicks + 0.30 * recall_carts + 0.60 * recall_orders

print(f"离线 Recall@20 (加权): {score:.5f}")
print(f"Clicks: {recall_clicks:.5f}, Carts: {recall_carts:.5f}, Orders: {recall_orders:.5f}")

submission_rows = []

with open(test_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Generating Test Recommendations"):
        session = json.loads(line)
        sid = session['session']
        preds = recommend(session)

        for t in ["clicks", "carts", "orders"]:
            submission_rows.append({
                "session_type": f"{sid}_{t}",
                "labels": " ".join(preds)
            })
submission = pd.DataFrame(submission_rows)
submission.to_csv('output\\submission.csv', index=False)
