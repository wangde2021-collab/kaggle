# NFL Big Data Bowl 2027 — Phase 1 ETL: CSV -> Parquet
#
# Per notes/gemini_scheme.md Stage 1:
#  - stream game_tracking_*.csv in chunks, semi-join games.csv (season_type == 'REG')
#    to drop the 147 preseason games (POST kept out as well; sensitivity uses REG+POST later)
#  - downcast float64 -> float32 for coordinates / speed / accel
#  - player_play.csv: RFC-4180 multiline-quoted parse, assert exactly 314,197 rows,
#    drop play_description text column (140 MB -> ~18 MB)
#  - write Zstd Parquet per file; emit data/parquet/manifest.json with row counts
#
# Usage:
#   python3 src/etl_parquet.py [--out data/parquet] [--limit-rows N]
# --limit-rows caps rows read from each game_tracking file (smoke test only).
import argparse
import json
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

SMALL_FILES = ["players.csv", "combine_results.csv", "games.csv",
               "player_career_successes.csv"]
GAME_TRACKING = ["game_tracking_2023.csv", "game_tracking_2024.csv",
                 "game_tracking_2025.csv"]
FLOAT32_COLS = ["x", "y", "s", "a", "dis", "o", "dir"]

EXPECTED_ROWS = {
    "players.csv": 510,
    "combine_results.csv": 510,
    "games.csv": 1002,
    "player_career_successes.csv": 510,
    "player_play.csv": 314197,
    "combine_tracking.csv": 463189,
    "game_tracking_2023.csv": 3665443,
    "game_tracking_2024.csv": 7827944,
    "game_tracking_2025.csv": 11160903,
}


def to_parquet(df, path):
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)


def etl_small(out):
    manifest = {}
    for name in SMALL_FILES:
        df = pd.read_csv(os.path.join(DATA, name))
        assert len(df) == EXPECTED_ROWS[name], f"{name}: {len(df)} != {EXPECTED_ROWS[name]}"
        dest = os.path.join(out, name.replace(".csv", ".parquet"))
        to_parquet(df, dest)
        manifest[name] = {"rows": len(df), "parquet": dest,
                          "mb": round(os.path.getsize(dest) / 1e6, 2)}
        print(f"{name}: {len(df):,} rows -> {dest} ({manifest[name]['mb']} MB)")
    return manifest


def etl_player_play(out):
    path = os.path.join(DATA, "player_play.csv")
    df = pd.read_csv(path)  # pandas C engine handles RFC-4180 multiline quotes
    assert len(df) == EXPECTED_ROWS["player_play.csv"], \
        f"player_play: {len(df)} != 314197 (check multiline-quote parsing)"
    n_text = df["play_description"].notna().sum()
    df = df.drop(columns=["play_description"])
    dest = os.path.join(out, "player_play.parquet")
    to_parquet(df, dest)
    print(f"player_play: {len(df):,} rows (dropped {n_text:,} text cells) -> "
          f"{dest} ({os.path.getsize(dest) / 1e6:.2f} MB)")
    return {"player_play.csv": {"rows": len(df), "parquet": dest,
                                "mb": round(os.path.getsize(dest) / 1e6, 2)}}


def etl_combine_tracking(out):
    path = os.path.join(DATA, "combine_tracking.csv")
    df = pd.read_csv(path)
    assert len(df) == EXPECTED_ROWS["combine_tracking.csv"], \
        f"combine_tracking: {len(df)} != 463189"
    dest = os.path.join(out, "combine_tracking.parquet")
    to_parquet(df, dest)
    print(f"combine_tracking: {len(df):,} rows -> {dest} "
          f"({os.path.getsize(dest) / 1e6:.2f} MB)")
    return {"combine_tracking.csv": {"rows": len(df), "parquet": dest,
                                     "mb": round(os.path.getsize(dest) / 1e6, 2)}}


def etl_game_tracking(out, limit_rows=None):
    games = pd.read_csv(os.path.join(DATA, "games.csv"))
    reg_ids = set(games.loc[games["season_type"] == "REG", "game_id"])
    manifest = {}
    for name in GAME_TRACKING:
        path = os.path.join(DATA, name)
        dest = os.path.join(out, name.replace(".csv", ".parquet"))
        total_in, total_out = 0, 0
        writer = None
        for chunk in pd.read_csv(path, chunksize=1_000_000):
            total_in += len(chunk)
            chunk = chunk[chunk["game_id"].isin(reg_ids)]
            for c in FLOAT32_COLS:
                if c in chunk.columns:
                    chunk[c] = chunk[c].astype("float32")
            total_out += len(chunk)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(dest, table.schema, compression="zstd")
            writer.write_table(table)
            if limit_rows and total_in >= limit_rows:
                break
        if writer is not None:
            writer.close()
        assert total_in == EXPECTED_ROWS[name] or (limit_rows and total_in >= limit_rows), \
            f"{name}: read {total_in} != {EXPECTED_ROWS[name]}"
        manifest[name] = {"rows_in": total_in, "rows_reg_only": total_out,
                          "parquet": dest,
                          "mb": round(os.path.getsize(dest) / 1e6, 2)}
        print(f"{name}: {total_in:,} rows in, {total_out:,} REG rows -> "
              f"{dest} ({manifest[name]['mb']} MB)")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "parquet"))
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="smoke test: cap rows read per game_tracking file")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest = {}
    manifest.update(etl_small(args.out))
    manifest.update(etl_player_play(args.out))
    manifest.update(etl_combine_tracking(args.out))
    manifest.update(etl_game_tracking(args.out, args.limit_rows))
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("manifest written:", os.path.join(args.out, "manifest.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
