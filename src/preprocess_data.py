"""
Step 2 — Text preprocessing.

Reads a raw (train | val | test) parquet produced by prepare_data.py and
optionally applies the French-language cleaning pipeline.

The output path encodes whether preprocessing was applied so downstream
steps can select the right artifact without ambiguity.
"""

import logging
import os
import sys

import hydra
from omegaconf import DictConfig
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
# Main Entrypoint with Hydra
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig) -> None:
    fs = get_fs()

    # Extraction des chemins passés dynamiquement (via Argo ou config)
    train_path = cfg.get("train_path")
    val_path = cfg.get("val_path")
    test_path = cfg.get("test_path")

    # Récupération de l'output_prefix par défaut si non fourni par la CLI
    output_prefix = cfg.get("output_prefix", "s3://mateom/graal/ttc-injection/preprocessed")

    # Gestion robuste du booléen (qu'il vienne d'Argo sous forme de string ou d'Hydra)
    raw_preprocessed = cfg.tokenizer.preprocessed
    if isinstance(raw_preprocessed, str):
        do_preprocess = raw_preprocessed.strip().lower() == "true"
    else:
        do_preprocess = bool(raw_preprocessed)

    # Validation minimale des arguments requis
    if not all([train_path, val_path, test_path]):
        logger.error("Missing required paths. Ensure train_path, val_path, and test_path are provided.")
        sys.exit(1)

    out_train = make_output_path(output_prefix, train_path, do_preprocess)
    out_val   = make_output_path(output_prefix, val_path,   do_preprocess)
    out_test  = make_output_path(output_prefix, test_path,  do_preprocess)

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

        process_and_write(train_path, out_train)
        process_and_write(val_path,   out_val)
        process_and_write(test_path,  out_test)

    # Emit paths for Argo
    print(f"TRAIN_PATH={out_train}")
    print(f"VAL_PATH={out_val}")
    print(f"TEST_PATH={out_test}")


if __name__ == "__main__":
    main()
