import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq
import gc
from tqdm import tqdm

train_features_path = "../dataset/train_features.parquet"
train_labels_path = "../dataset/train_labels.parquet"
batch_size = 50_000

def labels_generator(labels_path):
    parquet_file = pq.ParquetFile(labels_path)
    for i in range(parquet_file.num_row_groups):
        batch = parquet_file.read_row_group(i).to_pandas()
        session_dict = {}
        for _, row in batch.iterrows():
            session = row['session']
            session_dict[session] = {
                'clicks': set(row.get('clicks', [])),
                'carts': set(row.get('carts', [])),
                'orders': set(row.get('orders', []))
            }
        yield session_dict  # 逐步返回，一小批一小批的提供给训练过程
        del batch
        gc.collect()


def feature_batch_generator(features_path, labels_gen):
    parquet_file = pq.ParquetFile(features_path)
    for i in range(parquet_file.num_row_groups):
        features = parquet_file.read_row_group(i).to_pandas()
        try:
            session_dict = next(labels_gen)
        except StopIteration:
            session_dict = {}
        clicks_label, carts_label, orders_label = [], [], []
        for _, row in features.iterrows():
            s = row['session']
            c = row['candidate']
            labels = session_dict.get(s, {'clicks': set(), 'carts': set(), 'orders': set()})
            clicks_label.append(1 if c in labels['clicks'] else 0)
            carts_label.append(1 if c in labels['carts'] else 0)
            orders_label.append(1 if c in labels['orders'] else 0)
        features['clicks_label'] = clicks_label
        features['carts_label'] = carts_label
        features['orders_label'] = orders_label
        yield features
        del features, clicks_label, carts_label, orders_label
        gc.collect()


def train_lgb_incremental(features_path, labels_path, target_cols):
    labels_gen = labels_generator(labels_path)
    models = {target: None for target in target_cols}

    parquet_file = pq.ParquetFile(features_path)
    total_batches = parquet_file.num_row_groups

    for batch_idx, batch_df in enumerate(tqdm(feature_batch_generator(features_path, labels_gen),
                                              total=total_batches,
                                              desc="Processing Batches")):
        feature_cols = [c for c in batch_df.columns if c not in ['session', 'candidate',
                                                                 'clicks_label', 'carts_label', 'orders_label']]
        for target_col in tqdm(target_cols, desc=f"Training Models on Batch {batch_idx + 1}", leave=False):
            X = batch_df[feature_cols]
            y = batch_df[target_col]
            lgb_train = lgb.Dataset(X, label=y)

            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 64,
                'max_depth': -1,
                'min_data_in_leaf': 50,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbosity': -1
            }

            if models[target_col] is None:
                models[target_col] = lgb.train(params, lgb_train, num_boost_round=100)
            else:
                models[target_col] = lgb.train(params, lgb_train, num_boost_round=100, init_model=models[target_col])

            del X, y, lgb_train
            gc.collect()

        del batch_df
        gc.collect()

    # 保存模型
    for target_col, model in models.items():
        model.save_model(f"lgbm_{target_col}.txt")

    return models


if __name__ == "__main__":
    target_cols = ['clicks_label', 'carts_label', 'orders_label']
    models = train_lgb_incremental(train_features_path, train_labels_path, target_cols)
    print("All models trained and saved successfully.")
