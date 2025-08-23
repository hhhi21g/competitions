import json
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd

train_path = "dataset\\train.jsonl"
test_path = "dataset\\test.jsonl"
co_vis_pkl = "output\\co_visitation.pkl"
output_path = "output\\submission.csv"


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

# 构建商品共现字典
# with open(train_path, 'r', encoding='utf-8') as f:
#     for line in tqdm(f, total=total_lines, desc='构建共现矩阵'):
#         session = json.loads(line)
#         events = session['events']
#
#         # 更新全局热门
#         for e in events:
#             global_popular[e['aid']] += 1
#
#         aids = [event['aid'] for event in session['events'] if event['type'] == 'clicks']
#         if len(aids) < 2:  # 点击数太少则跳过
#             continue
#
#         # 只考虑相邻点击
#         for i in range(len(aids) - 1):
#             a, t1 = events[i]['aid'], events[i]['type']
#             b, t2 = events[i + 1]['aid'], events[i + 1]['type']
#             if a == b:
#                 continue
#
#             # clicks共现
#             if t1 == 'clicks' and t2 == 'clicks':
#                 co_visitation_clicks[a][b] += 1
#                 co_visitation_clicks[b][a] += 1
#
#             # click -> cart
#             if t1 == 'clicks' and t2 == 'carts':
#                 co_visitation_order[a][b] += 1
#
#             # click -> order
#             if t1 == 'clicks' and t2 == 'orders':
#                 co_visitation_order[a][b] += 1
# 构建商品共现字典（带时间戳处理）
with open(train_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, total=total_lines, desc='构建共现矩阵'):
        session = json.loads(line)
        events = session['events']

        # 按时间戳排序，避免乱序
        events = sorted(events, key=lambda x: x['ts'])

        # 更新全局热门
        for e in events:
            global_popular[e['aid']] += 1

        # 只保留点击商品序列
        clicks = [e for e in events if e['type'] == 'clicks']
        if len(clicks) < 2:  # 点击数太少则跳过
            continue

        # 遍历相邻点击
        for i in range(len(clicks) - 1):
            a, t1, ts1 = clicks[i]['aid'], clicks[i]['type'], clicks[i]['ts']
            b, t2, ts2 = clicks[i + 1]['aid'], clicks[i + 1]['type'], clicks[i + 1]['ts']
            if a == b:
                continue

            # 时间差（秒）
            dt = max(1, ts2 - ts1)
            # 时间权重：越近权重越高（可以调公式）
            w = 1.0 / (1 + (dt / 600))  # 600秒(10分钟)作缩放

            # clicks共现
            co_visitation_clicks[a][b] += w
            co_visitation_clicks[b][a] += w

        # 处理 click -> cart / order（保持原逻辑，但带时间权重）
        for i in range(len(events) - 1):
            a, t1, ts1 = events[i]['aid'], events[i]['type'], events[i]['ts']
            b, t2, ts2 = events[i + 1]['aid'], events[i + 1]['type'], events[i + 1]['ts']
            if a == b:
                continue

            dt = max(1, ts2 - ts1)
            w = 1.0 / (1 + (dt / 600))

            if t1 == 'clicks' and t2 == 'carts':
                co_visitation_cart[a][b] += w
            if t1 == 'clicks' and t2 == 'orders':
                co_visitation_order[a][b] += w


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
