import json
from collections import Counter
import pandas as pd
from tqdm import tqdm

top_clicks = Counter()

# 计算总行数
with open('dataSet\\train.jsonl', 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

with open('dataSet\\train.jsonl', 'r', encoding='utf-8') as f:
    for line in tqdm(f,total=total_lines,desc="统计热门商品"):
        session = json.loads(line)
        for event in session['events']:
            top_clicks[event['aid']] += 1

# 返回出现次数最多的前20个商品,返回的是一个列表(商品ID,出现次数)
top_items = [str(aid) for aid, _ in top_clicks.most_common(20)]

submission_rows = []

# 获取 test 文件总行数
with open('dataSet\\test.jsonl', 'r', encoding='utf-8') as f:
    total_test = sum(1 for _ in f)

with open('dataSet\\test.jsonl', 'r', encoding='utf-8') as f:
    for line in tqdm(f,total = total_test,desc="构建submission.csv"):
        session = json.loads(line)
        sid = session['session']
        for t in ['clicks', 'carts', 'orders']:
            submission_rows.append({
                'session_type': f"{sid}_{t}",
                'labels': ' '.join(top_items)
            })

submission = pd.DataFrame(submission_rows)
submission.to_csv('output\\submission.csv', index=False)