"""
Step 1 — Data preparation (Nested Sets with Guaranteed Nomenclature Cover Version).

For a given synth_path and a list of synth_splits, produce nested subdatasets
ensuring strict inclusion AND 100% code coverage across all splits.
"""

import hashlib
import logging
import os
import sys
from typing import List

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
        endpoint_url="https://minio.lab.sspcloud.fr",
        client_kwargs={"region_name": "us-east-1"},
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


def split_guaranteed_and_remaining(df: pl.DataFrame, code_column: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Sépare le dataset en deux :
    - Un dataframe garanti contenant exactement une ligne par code unique (mélangé proprement).
    - Un dataframe contenant le reste des lignes disponibles.
    """
    # On mélange d'abord pour ne pas toujours prendre la première ligne absolue du fichier source
    df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)
    
    df_guaranteed = df_shuffled.unique(subset=[code_column], keep="first", maintain_order=True)
    df_remaining = df_shuffled.join(df_guaranteed, on=df.columns, how="anti", maintain_order="left_right")
    
    return df_guaranteed, df_remaining


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

    raw_splits = cfg.injection.synth_splits
    if isinstance(raw_splits, str):
        synth_splits = [float(x.strip()) for x in raw_splits.split(",")]
    else:
        synth_splits = [float(x) for x in OmegaConf.to_object(raw_splits)]

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
    # Préparation des Réservoirs avec Nomenclature Garantie
    # ------------------------------------------------------------------
    logger.info("Splitting master data into guaranteed nomenclature bases and remaining pools …")
    
    # Réel / Original
    df_real_raw = fetch_original_data(cfg.data.original_train_path, fs)
    df_real_base, df_real_rem = split_guaranteed_and_remaining(df_real_raw, "code")
    logger.info(f"Real data: {len(df_real_base)} guaranteed codes, {len(df_real_rem)} remaining rows available.")

    # Synthétique
    with fs.open(synth_path) as f:
        df_synth_raw = pl.read_parquet(f)
    if "name" in df_synth_raw.columns:
        df_synth_raw = df_synth_raw.drop("name")
    df_synth_base, df_synth_rem = split_guaranteed_and_remaining(df_synth_raw, "code")
    logger.info(f"Synthetic data: {len(df_synth_base)} guaranteed codes, {len(df_synth_rem)} remaining rows available.")

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

        # Calcul du nombre de lignes cibles réelles et synthétiques globales
        synth_target_size = round(split * final_size)
        real_target_size = final_size - synth_target_size

        logger.info(f"Building nested train split: ratio={split} (synth target: {synth_target_size}, real target: {real_target_size})")

        # --- Bloc Synthétique ---
        if synth_target_size == 0:
            df_synth_chunk = pl.DataFrame(schema=df_synth_base.schema)
        elif synth_target_size <= len(df_synth_base):
            # Cas extrême ou échantillon cible plus petit que le nombre de codes uniques. 
            # On restreint la base fixe tout en restant déterministe.
            df_synth_chunk = df_synth_base.head(synth_target_size)
        else:
            # Cas classique : On prend toute la base nomenclature + le complément nécessaire depuis le reste
            rem_synth_needed = synth_target_size - len(df_synth_base)
            df_synth_chunk = pl.concat([df_synth_base, df_synth_rem.head(rem_synth_needed)])

        # --- Bloc Réel / Original ---
        if real_target_size == 0:
            df_real_chunk = pl.DataFrame(schema=df_real_base.schema)
        elif real_target_size <= len(df_real_base):
            df_real_chunk = df_real_base.head(real_target_size)
        else:
            rem_real_needed = real_target_size - len(df_real_base)
            df_real_chunk = pl.concat([df_real_base, df_real_rem.head(rem_real_needed)])

        # Validation finale des volumes
        total_extracted = len(df_real_chunk) + len(df_synth_chunk)
        if total_extracted != final_size:
            logger.error(f"Size mismatch for split {split}: got {total_extracted} rows instead of {final_size}")
            sys.exit(1)

        # Fusion des deux blocs et brassage final
        df_train = pl.concat([df_real_chunk, df_synth_chunk]).sample(fraction=1.0, shuffle=True, seed=42)
        s3_write(df_train, train_key, fs)

    # Output à destination d'Argo
    print(f"TRAIN_PATHS={','.join(train_paths_emitted)}")
    print(f"VAL_PATH={val_key}")
    print(f"TEST_PATH={test_key}")


if __name__ == "__main__":
    main()