import json
import pickle
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

train_path = "dataset/train.jsonl"
co_vis_pkl = "dataset_1/co_visitation.pkl"

# ===================== 参数设置 =====================
window = 3          # 滑动窗口大小
alpha = 0.001       # 时间衰减参数
topk = 50           # 每个商品保留 topk 共现商品

# ===================== 初始化 =====================
co_visitation_clicks = defaultdict(lambda: defaultdict(float))
co_visitation_cart = defaultdict(lambda: defaultdict(float))
co_visitation_order = defaultdict(lambda: defaultdict(float))

global_clicks = Counter()
global_carts = Counter()
global_orders = Counter()

# ===================== 统计总行数 =====================
def count_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

total_lines = count_lines(train_path)

# ===================== 构建共现矩阵 =====================
with open(train_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, total=total_lines, desc='构建共现矩阵'):
        session = json.loads(line)
        events = session['events']

        # 按时间排序
        events = sorted(events, key=lambda x: x['ts'])

        # 全局热门统计
        for e in events:
            if e['type'] == 'clicks':
                global_clicks[e['aid']] += 1
            elif e['type'] == 'carts':
                global_carts[e['aid']] += 1
            elif e['type'] == 'orders':
                global_orders[e['aid']] += 1

        # 滑动窗口统计共现
        for i in range(len(events)):
            a, t1, ts1 = events[i]['aid'], events[i]['type'], events[i]['ts']
            for j in range(i+1, min(i+1+window, len(events))):
                b, t2, ts2 = events[j]['aid'], events[j]['type'], events[j]['ts']
                if a == b:
                    continue

                # 时间差
                dt = max(1, ts2 - ts1)
                weight = float(np.exp(-alpha * dt))

                # 点击共现
                if t1 == 'clicks' and t2 == 'clicks':
                    co_visitation_clicks[a][b] += weight
                    co_visitation_clicks[b][a] += weight

                # 点击 -> 加购 / 购买
                if t1 == 'clicks' and t2 == 'carts':
                    co_visitation_cart[a][b] += weight
                if t1 == 'clicks' and t2 == 'orders':
                    co_visitation_order[a][b] += weight

# ===================== 只保留 topk =====================
def trim_topk(matrix, k=50):
    for a in matrix:
        matrix[a] = dict(sorted(matrix[a].items(), key=lambda x: -x[1])[:k])
    return matrix

co_visitation_clicks = trim_topk(co_visitation_clicks, topk)
co_visitation_cart = trim_topk(co_visitation_cart, topk)
co_visitation_order = trim_topk(co_visitation_order, topk)

# ===================== 保存 =====================
with open(co_vis_pkl, 'wb') as f:
    pickle.dump({
        "clicks": dict(co_visitation_clicks),
        "cart": dict(co_visitation_cart),
        "order": dict(co_visitation_order),
        "global_clicks": dict(global_clicks),
        "global_carts": dict(global_carts),
        "global_orders": dict(global_orders)
    }, f)

print(f'共现矩阵已保存到 {co_vis_pkl}')
