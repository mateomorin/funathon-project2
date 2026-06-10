"""
Step 2 — Text preprocessing.

Reads a raw (train | val | test) parquet produced by prepare_data.py and
optionally applies the French-language cleaning pipeline.

The output path encodes whether preprocessing was applied so downstream
steps can select the right artifact without ambiguity.
"""

import logging

import hydra
from omegaconf import DictConfig
import polars as pl
import s3fs
from dotenv import load_dotenv
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
french_stopwords = stopwords.words("french")

load_dotenv(override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        endpoint_url="https://minio.lab.sspcloud.fr",
        client_kwargs={"region_name": "us-east-1"},
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


def process_and_write(src_path: str, out_path: str, fs: s3fs.S3FileSystem):
    if not s3_exists(out_path, fs):
        df = s3_read(src_path, fs)
        df = preprocess(df, text_column="label", stopwords=french_stopwords)
        s3_write(df, out_path, fs)
        logger.info(f"Preprocessed {src_path} → {out_path}")
    else:
        logger.info(f"Already exists, skipping: {out_path}")


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def stem(s3_path: str) -> str:
    """Return the filename stem (no extension) of an S3 path."""
    return s3_path.rstrip("/").split("/")[-1].replace(".parquet", "")


# ---------------------------------------------------------------------------
# Main Entrypoint with Hydra
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../config", config_name="data_config")
def main(cfg: DictConfig) -> None:
    fs = get_fs()
    output_prefix = cfg.output_prefix
    train_folder = f"{output_prefix}/train/"
    shared_folder = f"{output_prefix}/shared/"

    for folder in [train_folder, shared_folder]:
        if not fs.exists(folder):
            continue

        files = fs.ls(folder)
        for file_path in files:
            if file_path.endswith(".parquet") and not file_path.endswith("_preprocessed.parquet"):
                preprocessed_path = file_path.replace(".parquet", "_preprocessed.parquet")
                if not s3_exists(preprocessed_path, fs):
                    process_and_write(file_path, preprocessed_path, fs)


if __name__ == "__main__":
    main()
