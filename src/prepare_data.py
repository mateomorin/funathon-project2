"""
Step 1 — Data preparation.

For a given (synth_path, synth_split, final_size) triplet, produce:
  - train split (mixed real + synthetic according to synth_split)
  - val   split (sampled once, shared across all experiments)
  - test  split (sampled once, shared across all experiments)

Outputs are written to S3 as parquet files so downstream steps never
re-read the raw sources.  Val/test paths are written under a
deterministic key that depends only on val_test_sample, so they are
produced only once even when this step is called many times.

Usage (called by Argo, one pod per synth_split value):
    python prepare_data.py \
        --synth_path s3://... \
        --synth_split 0.2 \
        --final_size  50000 \
        --val_test_sample 30000 \
        --output_prefix s3://projet-ape/ttc-injection/prepared \
        --original_train_path s3://... \
        --original_val_path   s3://... \
        --original_test_path  s3://...
"""

import argparse
import hashlib
import logging
import os
import sys

import polars as pl
import s3fs
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Helpers (same logic as original data.py)
# ---------------------------------------------------------------------------

def get_fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL", "https://minio.lab.sspcloud.fr"),
        client_kwargs={"region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")},
    )


def fetch_original_data(path, fs=None) -> pl.DataFrame:
    opener = fs.open if fs else open
    with opener(path) as f:
        df = pl.read_parquet(f)
    df = df.rename(mapping={"nace2025": "code", "libelle": "label"})[["code", "label"]]
    df = df.with_columns(
        (pl.col("code").str.slice(0, 2) + "." + pl.col("code").str.slice(2)).alias("code")
    )
    return df


def sample_with_all_codes(df: pl.DataFrame, code_column: str, sample_size: int) -> pl.DataFrame:
    df_guaranteed = df.unique(subset=[code_column], keep="first", maintain_order=True)
    df_remaining = df.join(df_guaranteed, on=df.columns, how="anti", maintain_order="left_right")
    remaining_size = sample_size - len(df_guaranteed)
    if remaining_size <= 0:
        logger.warning(
            f"sample_size={sample_size} < number of unique codes={len(df_guaranteed)}. "
            "Returning one row per code."
        )
        return df_guaranteed
    df_remaining_sampled = df_remaining.sample(n=remaining_size, seed=42)
    return pl.concat([df_guaranteed, df_remaining_sampled]).sample(fraction=1.0, seed=42)


def s3_write(df: pl.DataFrame, path: str, fs: s3fs.S3FileSystem) -> None:
    with fs.open(path, "wb") as f:
        df.write_parquet(f)
    logger.info(f"Written {len(df)} rows → {path}")


def s3_exists(path: str, fs: s3fs.S3FileSystem) -> bool:
    try:
        return fs.exists(path)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--synth_path", required=True)
    p.add_argument("--synth_split", type=float, required=True)
    p.add_argument("--final_size", type=int, required=True)
    p.add_argument("--val_test_sample", type=int, required=True)
    p.add_argument("--output_prefix", required=True,
                   help="S3 prefix, e.g. s3://projet-ape/ttc-injection/prepared")
    p.add_argument("--original_train_path", required=True)
    p.add_argument("--original_val_path",   required=True)
    p.add_argument("--original_test_path",  required=True)
    return p.parse_args()


def make_train_key(synth_path: str, synth_split: float, final_size: int) -> str:
    """Deterministic short key for this (synth_path, synth_split, final_size) combo."""
    raw = f"{synth_path}|{synth_split}|{final_size}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:8]
    synth_name = synth_path.rstrip("/").split("/")[-1].replace(".parquet", "")
    split_str = str(synth_split).replace(".", "p")
    return f"{synth_name}_split{split_str}_{digest}"


def main():
    args = parse_args()
    fs = get_fs()

    synth_split = args.synth_split
    assert 0.0 <= synth_split <= 1.0, "synth_split must be in [0, 1]"

    final_size  = args.final_size
    synth_size  = round(synth_split * final_size)
    real_size   = round((1 - synth_split) * final_size)

    # ------------------------------------------------------------------
    # Val / test  (shared across all train experiments)
    # Written once under a key that depends only on val_test_sample.
    # ------------------------------------------------------------------
    val_key  = f"{args.output_prefix}/shared/val_n{args.val_test_sample}.parquet"
    test_key = f"{args.output_prefix}/shared/test_n{args.val_test_sample}.parquet"

    if s3_exists(val_key, fs) and s3_exists(test_key, fs):
        logger.info("Val/test already exist — skipping.")
    else:
        logger.info("Computing shared val/test splits …")
        df_val  = fetch_original_data(args.original_val_path,  fs)
        df_test = fetch_original_data(args.original_test_path, fs)
        df_val  = df_val.sample(n=args.val_test_sample,  shuffle=True, seed=42)
        df_test = df_test.sample(n=args.val_test_sample, shuffle=True, seed=42)
        s3_write(df_val,  val_key,  fs)
        s3_write(df_test, test_key, fs)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    train_key_name = make_train_key(args.synth_path, synth_split, final_size)
    train_key = f"{args.output_prefix}/train/{train_key_name}.parquet"

    if s3_exists(train_key, fs):
        logger.info(f"Train artifact already exists at {train_key} — skipping.")
    else:
        logger.info(f"Building train split (synth_split={synth_split}, final_size={final_size}) …")

        if synth_split == 1.0:
            with fs.open(args.synth_path) as f:
                df_train = pl.read_parquet(f)
            if "name" in df_train.columns:
                df_train = df_train.drop("name")
            if final_size > len(df_train):
                logger.error(
                    f"Synthetic dataset ({len(df_train)} rows) is smaller than final_size={final_size}."
                )
                sys.exit(1)
            df_train = sample_with_all_codes(df_train, "code", final_size)

        elif synth_split == 0.0:
            df_train = fetch_original_data(args.original_train_path, fs)
            if final_size > len(df_train):
                logger.error(
                    f"Original train dataset ({len(df_train)} rows) is smaller than final_size={final_size}."
                )
                sys.exit(1)
            df_train = sample_with_all_codes(df_train, "code", final_size)

        else:
            # Mixed
            with fs.open(args.synth_path) as f:
                df_synth = pl.read_parquet(f)
            if "name" in df_synth.columns:
                df_synth = df_synth.drop("name")
            if synth_size > len(df_synth):
                logger.error(
                    f"Synthetic dataset ({len(df_synth)} rows) is smaller than synth_size={synth_size}."
                )
                sys.exit(1)
            df_synth = sample_with_all_codes(df_synth, "code", synth_size)

            df_real = fetch_original_data(args.original_train_path, fs)
            if real_size > len(df_real):
                logger.error(
                    f"Real train dataset ({len(df_real)} rows) is smaller than real_size={real_size}."
                )
                sys.exit(1)
            df_real = sample_with_all_codes(df_real, "code", real_size)

            df_train = pl.concat([df_real, df_synth]).sample(fraction=1.0, shuffle=True, seed=42)

        s3_write(df_train, train_key, fs)

    # ------------------------------------------------------------------
    # Emit output paths for Argo (printed to stdout, one per line)
    # ------------------------------------------------------------------
    print(f"TRAIN_PATH={train_key}")
    print(f"VAL_PATH={val_key}")
    print(f"TEST_PATH={test_key}")


if __name__ == "__main__":
    main()