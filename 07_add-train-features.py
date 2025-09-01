# 07_add-train-features_fixed.py
import os
import json
import sqlite3
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ========== CONFIG ==========
CANDIDATES_PATH = "dataset/candidates.parquet"
TEST_PATH = "dataset/train.jsonl"
CO_PKL_PATH = "dataset_1/co_visitation.pkl"
OUTPUT_PATH = "dataset_1/train_features.parquet"
SESSION_DB = "dataset_1/session_kv.sqlite"

CHUNK_SIZE = 1_000_000    # rows read from candidates.parquet per iteration
BATCH_WRITE = 500_000     # write features to parquet every this many rows
ALPHA = 0.001             # time decay
SQL_IN_CHUNK = 2000       # sqlite IN chunk size

# ========== helpers: co_vis load (normalize keys to int) ==========
def load_co_vis_normalized(co_pkl_path):
    with open(co_pkl_path, "rb") as f:
        co_vis_raw = pickle.load(f)
    # if wrapped under "clicks"
    if isinstance(co_vis_raw, dict) and "clicks" in co_vis_raw:
        co_vis_raw = co_vis_raw["clicks"]
    co_vis = {}
    for k, v in co_vis_raw.items():
        try:
            ik = int(k)
        except Exception:
            ik = k
        if isinstance(v, dict):
            inner = {}
            for kk, vv in v.items():
                try:
                    ikk = int(kk)
                except Exception:
                    ikk = kk
                inner[ikk] = float(vv)
            co_vis[ik] = inner
        else:
            # if not dict, try to cast to dict
            try:
                co_vis[ik] = dict(v)
            except Exception:
                co_vis[ik] = {}
    return co_vis

# ========== helpers: fetch sessions from sqlite ==========
def fetch_sessions_from_db(sqlite_path, session_list):
    """
    Given a list of session ids (any type), query sqlite 'sessions' table where
    we stored (session TEXT, aids_json, ts_json).
    Returns dict: session_str -> (np.array(aids,int64), np.array(ts,int64))
    """
    if len(session_list) == 0:
        return {}

    # convert to strings (sqlite stored session as TEXT in your build step)
    sessions = [str(s) for s in session_list]
    session_map = {}
    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()
    for i in range(0, len(sessions), SQL_IN_CHUNK):
        chunk = sessions[i:i + SQL_IN_CHUNK]
        placeholders = ",".join("?" for _ in chunk)
        sql = f"SELECT session, aids_json, ts_json FROM sessions WHERE session IN ({placeholders})"
        try:
            for row in c.execute(sql, chunk):
                sess, aids_json, ts_json = row
                # parse JSON arrays
                try:
                    aids_list = json.loads(aids_json)
                    ts_list = json.loads(ts_json)
                    aids = np.array(aids_list, dtype=np.int64) if len(aids_list) > 0 else np.array([], dtype=np.int64)
                    tss = np.array(ts_list, dtype=np.int64) if len(ts_list) > 0 else np.array([], dtype=np.int64)
                    session_map[str(sess)] = (aids, tss)
                except Exception:
                    # fallback: empty arrays when parse fails
                    session_map[str(sess)] = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        except Exception:
            # in case of unexpected SQL error, continue
            continue
    conn.close()
    return session_map

# ========== feature computation per batch (matrixized per session) ==========
def process_batch_df(df_batch, co_vis, sqlite_path, alpha=ALPHA):
    """
    df_batch: pandas.DataFrame, must contain 'session','candidate' (and optional 'score')
    co_vis: dict[int] -> dict[int]->float
    returns list of dicts (feature rows)
    """
    out_rows = []

    # ensure session column is string, to match sqlite TEXT keys
    df_batch = df_batch.copy()
    df_batch["session"] = df_batch["session"].astype(str)

    session_ids = df_batch["session"].unique().tolist()
    session_map = fetch_sessions_from_db(sqlite_path, session_ids)

    # group by session to reuse session-level arrays
    grouped = df_batch.groupby("session")

    for session_str, grp in grouped:
        # lookup by string
        sess_key = str(session_str)
        if sess_key not in session_map:
            # no session events found -> skip but continue progress
            continue
        aids, tss = session_map[sess_key]
        if aids.size == 0:
            continue

        session_len = aids.shape[0]
        max_ts = int(tss.max()) if tss.size > 0 else 0

        # prepare co dicts for each aid in session (list of dicts)
        co_dicts = [co_vis.get(int(a), {}) for a in aids]

        # candidates for this session (keep original order)
        cands = grp["candidate"].values.astype(np.int64)
        base_scores = grp["score"].values.astype(np.float32) if "score" in grp.columns else np.zeros(len(cands), dtype=np.float32)

        # build co_scores_matrix: shape (session_len, n_cands)
        # We'll fill row by row (session length is usually small), this is faster than inner Python loops across candidates
        n_cands = cands.shape[0]
        co_scores_matrix = np.zeros((session_len, n_cands), dtype=np.float32)
        for i, d in enumerate(co_dicts):
            # try to vectorize lookup for this row: for each candidate, get d.get(cand, 0.0)
            # can't fully vectorize dict lookup, do python loop across candidates but session_len typically small
            row_vals = np.empty(n_cands, dtype=np.float32)
            for j, cand in enumerate(cands):
                row_vals[j] = d.get(int(cand), 0.0)
            co_scores_matrix[i, :] = row_vals

        # statistics across session axis (axis=0 gives per-candidate stats)
        co_max = co_scores_matrix.max(axis=0)
        co_mean = co_scores_matrix.mean(axis=0)
        co_sum = co_scores_matrix.sum(axis=0)

        # time-weighted
        # weights shape (session_len,)
        weights = np.exp(-alpha * (max_ts - tss)).astype(np.float32)
        co_scores_weighted = co_scores_matrix * weights[:, None]
        co_max_w = co_scores_weighted.max(axis=0)
        co_mean_w = co_scores_weighted.mean(axis=0)
        co_sum_w = co_scores_weighted.sum(axis=0)

        candidate_in_session = np.isin(cands, aids).astype(np.int8)
        # candidate_last_ts: for each cand, if in session get last timestamp, else 0
        candidate_last_ts = np.zeros(n_cands, dtype=np.int64)
        for idx, cand in enumerate(cands):
            mask = (aids == cand)
            if mask.any():
                candidate_last_ts[idx] = int(tss[mask].max())
            else:
                candidate_last_ts[idx] = 0
        time_diff_last = np.where(candidate_last_ts > 0, max_ts - candidate_last_ts, -1)

        last_aid = int(aids[-1])
        last_ts = int(tss[-1])
        co_last = np.array([co_vis.get(last_aid, {}).get(int(cand), 0.0) for cand in cands], dtype=np.float32)
        last_click_match = (cands == last_aid).astype(np.int8)
        co_std = co_scores_matrix.std(axis=0)
        co_last_weighted = co_last * np.exp(-alpha * (max_ts - last_ts))

        ts_min = int(tss.min())
        duration = int(max_ts - ts_min) if session_len > 1 else 0
        avg_interval = float(duration / (session_len - 1)) if session_len > 1 else 0.0

        # assemble rows
        for i in range(n_cands):
            out_rows.append({
                "session": sess_key,
                "candidate": int(cands[i]),
                "base_score": float(base_scores[i]),
                "co_max": float(co_max[i]),
                "co_mean": float(co_mean[i]),
                "co_sum": float(co_sum[i]),
                "co_max_weighted": float(co_max_w[i]),
                "co_mean_weighted": float(co_mean_w[i]),
                "co_sum_weighted": float(co_sum_w[i]),
                "session_len": int(session_len),
                "session_duration": int(duration),
                "session_avg_interval": float(avg_interval),
                "candidate_in_session": int(candidate_in_session[i]),
                "candidate_last_ts": int(candidate_last_ts[i]),
                "time_diff_last": int(time_diff_last[i]),
                "last_click_match": int(last_click_match[i]),
                "co_last": float(co_last[i]),
                "co_last_weighted": float(co_last_weighted[i]),
                "co_std": float(co_std[i]),
            })
    return out_rows

# ========== main ==========
def main():
    # NOTE: do not force rebuild sqlite by default (it takes time).
    # If you change TRAIN jsonl, set force_rebuild=True below.
    if not os.path.exists(SESSION_DB):
        print("Session sqlite doesn't exist, building it...")
    else:
        print("Session sqlite exists, will use it.")

    # ensure session sqlite exists (build if missing)
    # set force_rebuild=True if you need to rebuild
    from_build = False
    if not os.path.exists(SESSION_DB):
        build_session = True
    else:
        build_session = False

    if build_session:
        # Reuse build function from your previous script (create sessions table with aids_json & ts_json)
        print("Building session sqlite DB (this may take a while)...")
        # minimal rebuild function inline to avoid imports mismatch
        conn = sqlite3.connect(SESSION_DB)
        conn.close()
        # call your external builder (assumes you have that function available)
        # Here we will call the previously provided build_session_sqlite (if present)
        try:
            # try to import function if user has script; else we run a quick inline builder
            from_build = True
            # fallback: user should create the DB beforehand; if not, script will fail
            print("Please ensure SESSION_DB exists. Exiting.")
            return
        except Exception:
            pass

    print("Loading co-visitation...")
    co_vis = load_co_vis_normalized(CO_PKL_PATH)
    print("co_vis loaded. co_vis keys:", len(co_vis))

    pf = pq.ParquetFile(CANDIDATES_PATH)
    total_rows = pf.metadata.num_rows if pf.metadata is not None else None
    pbar = tqdm(total=total_rows, desc="Overall", unit="rows")

    writer = None
    buffer_feat = []
    rows_written = 0

    # iterate candidates file in chunks
    for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
        df_batch = batch.to_pandas()
        # ensure required columns exist
        if "session" not in df_batch.columns or "candidate" not in df_batch.columns:
            # skip invalid batch
            pbar.update(len(df_batch))
            continue
        if "score" not in df_batch.columns:
            df_batch["score"] = 0.0

        # compute features for this batch
        res = process_batch_df(df_batch, co_vis, SESSION_DB, ALPHA)
        # append to buffer and write in big batches
        if res:
            buffer_feat.extend(res)

        # update overall progress by number of candidates processed (not by produced feature rows)
        pbar.update(len(df_batch))

        while len(buffer_feat) >= BATCH_WRITE:
            to_write = buffer_feat[:BATCH_WRITE]
            df_out = pd.DataFrame(to_write)
            table = pa.Table.from_pandas(df_out)
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_PATH, table.schema, compression="snappy")
            writer.write_table(table)
            buffer_feat = buffer_feat[BATCH_WRITE:]
            rows_written += len(to_write)
            print(f"Written {rows_written} rows so far to {OUTPUT_PATH}")

    # final flush
    if buffer_feat:
        df_out = pd.DataFrame(buffer_feat)
        table = pa.Table.from_pandas(df_out)
        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_PATH, table.schema, compression="snappy")
        writer.write_table(table)
        rows_written += len(buffer_feat)
        print(f"Final write: total rows = {rows_written}")

    if writer:
        writer.close()
    pbar.close()
    print(f"Done. Wrote {rows_written} feature rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
