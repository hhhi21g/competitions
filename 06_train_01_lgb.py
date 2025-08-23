import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# ============ Step 1. Session 层面划分 ============
# 获取所有 session id
all_sessions = train_features['session'].unique()

# 按 session 划分 80/20
train_sessions, valid_sessions = train_test_split(all_sessions, test_size=0.2, random_state=42)

# 根据 session id 过滤数据
train_df = train_features[train_features['session'].isin(train_sessions)]
valid_df = train_features[train_features['session'].isin(valid_sessions)]

print("Train size:", len(train_df))
print("Valid size:", len(valid_df))

# ============ Step 2. 准备特征和标签 ============
# 假设 "label" 是目标变量
y_train = train_df['label']
y_valid = valid_df['label']

# 去掉非特征列
drop_cols = ['session', 'item', 'label']  # 保留的都是数值特征
X_train = train_df.drop(columns=drop_cols)
X_valid = valid_df.drop(columns=drop_cols)

# ============ Step 3. 训练 LightGBM 模型 ============
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_valid, label=y_valid)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

model = lgb.train(
    params,
    train_set,
    valid_sets=[train_set, valid_set],
    num_boost_round=200,
    early_stopping_rounds=20,
    verbose_eval=20
)

# ============ Step 4. 验证集评估 ============
y_pred_proba = model.predict(X_valid, num_iteration=model.best_iteration)
y_pred = (y_pred_proba > 0.5).astype(int)

print("Validation AUC:", roc_auc_score(y_valid, y_pred_proba))
print("Validation ACC:", accuracy_score(y_valid, y_pred))
