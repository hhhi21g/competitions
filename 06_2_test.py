import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq
from tqdm import tqdm

test_features_path = "dataset/test_features.parquet"

# ========== 加载模型 ==========
models = {
    'clicks': lgb.Booster(model_file="lgbm_clicks_label.txt"),
    'carts':  lgb.Booster(model_file="lgbm_carts_label.txt"),
    'orders': lgb.Booster(model_file="lgbm_orders_label.txt")
}

# ========== 预测函数 ==========
def predict_submission(features_path, models, topk=20):
    parquet_file = pq.ParquetFile(features_path)
    print(f"[INFO] Total row groups in parquet: {parquet_file.num_row_groups}")

    results = []

    for rg in tqdm(range(parquet_file.num_row_groups), desc="Predicting"):
        batch = parquet_file.read_row_group(rg).to_pandas()
        print(f"[DEBUG] Row group {rg}: shape = {batch.shape}")

        batch_results = []  # 每个 batch 的结果先放临时列表

        for target in ["clicks", "carts", "orders"]:
            model = models[target]
            model_feature_names = model.feature_name()

            # 补齐缺失特征
            for f in model_feature_names:
                if f not in batch.columns:
                    batch[f] = 0

            X = batch[model_feature_names]
            batch[f"{target}_pred"] = model.predict(X)

            # 分组取 TopK
            preds = (
                batch.groupby("session")
                .apply(lambda x: x.nlargest(topk, f"{target}_pred")["candidate"].astype(str).tolist())
            )
            preds = preds.reset_index().rename(columns={0: "labels"})
            preds["labels"] = preds["labels"].apply(lambda x: " ".join(x))
            preds["session_type"] = preds["session"].astype(str) + f"_{target}"

            batch_results.append(preds[["session_type", "labels"]])

        # 合并并去重，避免 batch 内部重复
        batch_results = pd.concat(batch_results, ignore_index=True)
        batch_results = batch_results.drop_duplicates(subset=["session_type"], keep="first")

        results.append(batch_results)

    # 拼接所有 batch
    submission = pd.concat(results, ignore_index=True)

    # 再次全局去重，保证唯一
    submission = submission.drop_duplicates(subset=["session_type"], keep="first")

    return submission

# ========== 主程序 ==========
if __name__ == "__main__":
    submission = predict_submission(test_features_path, models, topk=20)
    submission.to_csv("submission.csv", index=False)
    print("✅ submission.csv saved successfully. Shape:", submission.shape)
    print(submission.head(6))
