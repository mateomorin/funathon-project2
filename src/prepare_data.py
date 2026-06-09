"""
Step 1 — Data preparation (Nested Sets Version).

For a given synth_path and a list of synth_splits, produce nested subdatasets
ensuring strict inclusion to guarantee statistical monotonicity.
"""

import hashlib
import logging
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
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


def fetch_original_data(path: str, fs=None) -> pl.DataFrame:
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


def make_train_key(synth_path: str, synth_split: float, final_size: int) -> str:
    """Deterministic short key for this (synth_path, synth_split, final_size) combo."""
    raw = f"{synth_path}|{synth_split}|{final_size}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:8]
    synth_name = synth_path.rstrip("/").split("/")[-1].replace(".parquet", "")
    split_str = str(synth_split).replace(".", "p")
    return f"{synth_name}_split{split_str}_{digest}"


# ---------------------------------------------------------------------------
# Main Entrypoint with Hydra
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig) -> None:
    fs = get_fs()

    # Extraction des paramètres
    synth_path = cfg.data.synth_path
    final_size = int(cfg.injection.final_size)
    val_test_sample = int(cfg.injection.val_test_sample)
    output_prefix = cfg.get("output_prefix", "s3://mateom/graal/ttc-injection/")

    # Gestion dynamique du type pour accepter une liste brute ou une string CSV
    raw_splits = cfg.injection.synth_splits
    if isinstance(raw_splits, str):
        synth_splits = [float(x.strip()) for x in raw_splits.split(",")]
    else:
        synth_splits = [float(x) for x in OmegaConf.to_object(raw_splits)]

    # Sorting pour s'assurer que notre logique d'incrémentation visuelle est propre
    synth_splits = sorted(synth_splits)

    # ------------------------------------------------------------------
    # Val / test (Inchangé, partagé globalement)
    # ------------------------------------------------------------------
    val_key = f"{output_prefix}/shared/val_n{val_test_sample}.parquet"
    test_key = f"{output_prefix}/shared/test_n{val_test_sample}.parquet"

    if s3_exists(val_key, fs) and s3_exists(test_key, fs):
        logger.info("Shared val/test already exist — skipping.")
    else:
        logger.info("Computing shared val/test splits …")
        df_val = fetch_original_data(cfg.data.original_val_path, fs)
        df_test = fetch_original_data(cfg.data.original_test_path, fs)
        df_val = df_val.sample(n=val_test_sample, shuffle=True, seed=42)
        df_test = df_test.sample(n=val_test_sample, shuffle=True, seed=42)
        s3_write(df_val, val_key, fs)
        s3_write(df_test, test_key, fs)

    # ------------------------------------------------------------------
    # Génération des Réservoirs de Base Fixes (Le cœur de l'inclusion)
    # ------------------------------------------------------------------
    logger.info(f"Sampling master pools of size final_size={final_size} …")

    # Pool 1 : Données originales
    df_real_raw = fetch_original_data(cfg.data.original_train_path, fs)
    if final_size > len(df_real_raw):
        logger.error(f"Original dataset ({len(df_real_raw)} rows) is smaller than final_size={final_size}.")
        sys.exit(1)
    df_real_pool = sample_with_all_codes(df_real_raw, "code", final_size)

    # Pool 2 : Données synthétiques
    with fs.open(synth_path) as f:
        df_synth_raw = pl.read_parquet(f)
    if "name" in df_synth_raw.columns:
        df_synth_raw = df_synth_raw.drop("name")
    if final_size > len(df_synth_raw):
        logger.error(f"Synthetic dataset ({len(df_synth_raw)} rows) is smaller than final_size={final_size}.")
        sys.exit(1)
    df_synth_pool = sample_with_all_codes(df_synth_raw, "code", final_size)

    # ------------------------------------------------------------------
    # Boucle de création des datasets imbriqués
    # ------------------------------------------------------------------
    train_paths_emitted = []

    for split in synth_splits:
        assert 0.0 <= split <= 1.0, f"synth_split {split} must be in [0, 1]"

        train_key_name = make_train_key(synth_path, split, final_size)
        train_key = f"{output_prefix}/train/{train_key_name}.parquet"
        train_paths_emitted.append(train_key)

        if s3_exists(train_key, fs):
            logger.info(f"Train artifact for split={split} already exists — skipping.")
            continue

        synth_size = round(split * final_size)
        real_size = final_size - synth_size

        logger.info(f"Building nested train split: ratio={split} (synth: {synth_size}, real: {real_size})")

        # Slice déterministe sur les réservoirs fixes pour garantir l'inclusion
        df_synth_chunk = df_synth_pool.head(synth_size)
        df_real_chunk = df_real_pool.head(real_size)

        # Fusion et sauvegarde
        df_train = pl.concat([df_real_chunk, df_synth_chunk]).sample(fraction=1.0, shuffle=True, seed=42)
        s3_write(df_train, train_key, fs)

    # ------------------------------------------------------------------
    # Adaptation Argo : On renvoie les variables attendues
    # ------------------------------------------------------------------
    # Comme le script traite TOUT d'un coup, on affiche la liste des chemins créés
    print(f"TRAIN_PATHS={','.join(train_paths_emitted)}")
    print(f"VAL_PATH={val_key}")
    print(f"TEST_PATH={test_key}")


if __name__ == "__main__":
    main()