"""
Step 2 — Text preprocessing.

Reads a raw (train | val | test) parquet produced by prepare_data.py and
optionally applies the French-language cleaning pipeline.

The output path encodes whether preprocessing was applied so downstream
steps can select the right artifact without ambiguity.

Usage:
    python preprocess_data.py \
        --train_path  s3://.../train/xxx.parquet \
        --val_path    s3://.../shared/val_n30000.parquet \
        --test_path   s3://.../shared/test_n30000.parquet \
        --preprocessed true \
        --output_prefix s3://projet-ape/ttc-injection/preprocessed
"""

import argparse
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
# Helpers
# ---------------------------------------------------------------------------

def get_fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL", "https://minio.lab.sspcloud.fr"),
        client_kwargs={"region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")},
    )


def s3_read(path: str, fs: s3fs.S3FileSystem) -> pl.DataFrame:
    with fs.open(path) as f:
        return pl.read_parquet(f)


def s3_write(df: pl.DataFrame, path: str, fs: s3fs.S3FileSystem) -> None:
    with fs.open(path, "wb") as f:
        df.write_parquet(f)
    logger.info(f"Written {len(df)} rows → {path}")


def s3_exists(path: str, fs: s3fs.S3FileSystem) -> bool:
    try:
        return fs.exists(path)
    except Exception:
        return False


def preprocess(df: pl.DataFrame, text_column: str, stopwords: list) -> pl.DataFrame:
    stopwords_pattern = r"\b(" + "|".join(stopwords) + r")\b"
    cleaned_expr = (
        pl.col(text_column)
        .cast(pl.String)
        .str.to_lowercase()
        .str.replace_all(r"[éèêë]", "e")
        .str.replace_all(r"[àâä]", "a")
        .str.replace_all(r"[ùûü]", "u")
        .str.replace_all(r"[îï]", "i")
        .str.replace_all(r"[ôö]", "o")
        .str.replace_all(r"[ç]", "c")
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(stopwords_pattern, " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )
    return df.with_columns(cleaned_expr)


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def stem(s3_path: str) -> str:
    """Return the filename stem (no extension) of an S3 path."""
    return s3_path.rstrip("/").split("/")[-1].replace(".parquet", "")


def make_output_path(prefix: str, original_path: str, preprocessed: bool) -> str:
    tag = "preprocessed" if preprocessed else "raw"
    name = stem(original_path)
    return f"{prefix}/{tag}/{name}.parquet"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path",      required=True)
    p.add_argument("--val_path",        required=True)
    p.add_argument("--test_path",       required=True)
    p.add_argument("--preprocessed",    required=True,
                   help="'true' or 'false'")
    p.add_argument("--output_prefix",   required=True,
                   help="S3 prefix, e.g. s3://projet-ape/ttc-injection/preprocessed")
    return p.parse_args()


def main():
    args = parse_args()
    do_preprocess = args.preprocessed.strip().lower() == "true"
    fs = get_fs()

    out_train = make_output_path(args.output_prefix, args.train_path, do_preprocess)
    out_val   = make_output_path(args.output_prefix, args.val_path,   do_preprocess)
    out_test  = make_output_path(args.output_prefix, args.test_path,  do_preprocess)

    # Val/test are shared: if already preprocessed by a sibling task, reuse.
    need_train = not s3_exists(out_train, fs)
    need_val   = not s3_exists(out_val,   fs)
    need_test  = not s3_exists(out_test,  fs)

    if not (need_train or need_val or need_test):
        logger.info("All outputs already exist — skipping.")
    else:
        if do_preprocess:
            import nltk
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            french_stopwords = stopwords.words("french")
            logger.info(f"Preprocessing enabled ({len(french_stopwords)} French stopwords).")
        else:
            french_stopwords = None
            logger.info("Preprocessing disabled — passing data through as-is.")

        def process_and_write(src_path: str, out_path: str):
            if not s3_exists(out_path, fs):
                df = s3_read(src_path, fs)
                if do_preprocess:
                    df = preprocess(df, text_column="label", stopwords=french_stopwords)
                s3_write(df, out_path, fs)
            else:
                logger.info(f"Already exists, skipping: {out_path}")

        process_and_write(args.train_path, out_train)
        process_and_write(args.val_path,   out_val)
        process_and_write(args.test_path,  out_test)

    # Emit paths for Argo
    print(f"TRAIN_PATH={out_train}")
    print(f"VAL_PATH={out_val}")
    print(f"TEST_PATH={out_test}")


if __name__ == "__main__":
    main()