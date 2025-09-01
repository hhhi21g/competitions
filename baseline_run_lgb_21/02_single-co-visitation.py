import json
import pickle
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

train_path = "../dataset/train.jsonl"
test_path = "../dataset/test.jsonl"
co_vis_pkl = "co_visitation.pkl" 
output_path = "submission.csv" 


def count_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

total_lines = count_lines(train_path)
total_test = count_lines(test_path)

# 嵌套字典: 字典里的值还是一个字典
# 外层字典: co_visitation[a], 商品a
# 内层字典: co_visitation[a][b], 表示商品a和商品b的共现次数
# defaultdict: 自动初始化
co_visitation = defaultdict(lambda: defaultdict(int))

# 构建商品共现字典
with open(train_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, total=total_lines, desc='构建共现矩阵'):
        session = json.loads(line)
        aids = [event['aid'] for event in session['events'] if event['type'] == 'clicks']
        if len(aids) < 2:  # 点击数太少则跳过
            continue

        # 只考虑相邻点击
        for i in range(len(aids) - 1):
            a, b = aids[i], aids[i + 1]
            if a != b:
                co_visitation[b][a] += 1
                co_visitation[a][b] += 1

# 每个商品只保留 top 50 共现商品
for a in co_visitation:
    co_visitation[a] = dict(sorted(co_visitation[a].items(), key=lambda x: -x[1])[:50])

# 保存共现矩阵(以二进制写入模式打开文件,适用于pickle序列化)
with open(co_vis_pkl,'wb') as f:
    pickle.dump(dict(co_visitation), f)

print(f'共现矩阵已保存到 {co_vis_pkl}')


submission_rows = [] 

with open(test_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, total=total_test, desc='为测试集生成推荐'):
        session = json.loads(line)
        sid = session['session']

        # 获得最近的点击事件
        clicked = [event['aid'] for event in session['events'] if event['type'] == 'clicks']
        recent_clicks = clicked[-5:]

        # 推荐商品池
        candidates = defaultdict(int)
        for aid in recent_clicks:
            related = co_visitation.get(aid, {})
            for b, score in related.items():
                candidates[b] += score  # 累加得分

        top_items = [str(aid) for aid, _ in sorted(candidates.items(), key=lambda x: -x[1])[:20]]

        for t in ['clicks', 'carts', 'orders']:
            submission_rows.append({
                'session_type': f"{sid}_{t}",
                'labels': ' '.join(top_items)
            })

submission = pd.DataFrame(submission_rows)
submission.to_csv('output\\submission.csv', index=False)