"""
Step 1 — Data preparation (Nested Sets with Guaranteed Nomenclature Cover Version).

For a given synth_path and a list of synth_splits, produce nested subdatasets
ensuring strict inclusion AND 100% code coverage across all splits.
"""

import logging

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
    return df


def fetch_synth_data(path: str, fs=None) -> pl.DataFrame:
    opener = fs.open if fs else open
    with opener(path) as f:
        df = pl.read_parquet(f)
    if "name" in df.columns:
        df = df.drop("name")
    return df


def split_guaranteed_and_remaining(
    df: pl.DataFrame,
    code_column: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Sépare le dataset en deux :
    - Un dataframe garanti contenant exactement une ligne par code unique (mélangé proprement).
    - Un dataframe contenant le reste des lignes disponibles.
    """
    # On mélange d'abord pour ne pas toujours prendre la première ligne absolue du fichier source
    df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)

    df_guaranteed = df_shuffled.unique(subset=[code_column], keep="first", maintain_order=True)
    df_remaining = df_shuffled.join(
        df_guaranteed,
        on=df.columns,
        how="anti",
        maintain_order="left_right"
    )

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


def retrieve_splits(raw_splits):
    if isinstance(raw_splits, str):
        synth_splits = [float(x.strip()) for x in raw_splits.split(",")]
    else:
        synth_splits = [float(x) for x in OmegaConf.to_object(raw_splits)]

    return sorted(synth_splits)


def make_train_key(output_prefix: str, synth_name: str, synth_split: float, final_size: int) -> str:
    """Format name_nXXXX_splitXpXX"""
    split_str = str(synth_split).replace(".", "p")
    output_prefix = output_prefix.rstrip("/")
    return f"{output_prefix}/train/{synth_name}_n{final_size}_split{split_str}.parquet"


# ---------------------------------------------------------------------------
# Main Entrypoint with Hydra
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="", config_name="data_config")
def main(cfg: DictConfig) -> None:
    fs = get_fs()

    # Extraction des paramètres
    synth_paths = OmegaConf.to_object(cfg.synth_paths)
    synth_names = OmegaConf.to_object(cfg.synth_names)
    synth_splits = retrieve_splits(cfg.synth_splits)
    final_size = int(cfg.final_size)
    output_prefix = cfg.output_prefix

    # ------------------------------------------------------------------
    # Préparation des Réservoirs avec Nomenclature Garantie
    # ------------------------------------------------------------------
    logger.info("Splitting master data into guaranteed nomenclature bases and remaining pools …")

    # Réel / Original
    df_real_raw = fetch_original_data(f"{output_prefix}/shared/train_n{final_size}.parquet", fs)
    df_real_base, df_real_rem = split_guaranteed_and_remaining(df_real_raw, "code")
    logger.info(f"Real data: {len(df_real_base)} guaranteed codes, {len(df_real_rem)} remaining rows available.")

    # Synthétique
    for synth_path, synth_name in zip(synth_paths, synth_names):
        df_synth_raw = fetch_synth_data(synth_path, fs)
        df_synth_base, df_synth_rem = split_guaranteed_and_remaining(df_synth_raw, "code")
        logger.info(f"Synthetic data: {len(df_synth_base)} guaranteed codes, {len(df_synth_rem)} remaining rows available.")

        # ------------------------------------------------------------------
        # Boucle de création des datasets imbriqués
        # ------------------------------------------------------------------
        train_paths_emitted = []

        for split in synth_splits:
            assert 0.0 <= split <= 1.0, f"synth_split {split} must be in [0, 1]"

            train_key = make_train_key(output_prefix, synth_name, split, final_size)
            train_paths_emitted.append(train_key)

            if s3_exists(train_key, fs):
                logger.info(f"Train artifact for split={split} already exists — skipping.")
                continue

            # Calcul du nombre de lignes cibles réelles et synthétiques globales
            synth_size = round(split * final_size)
            real_size = final_size - synth_size

            logger.info(f"Building nested train split: ratio={split} (synth target: {synth_size}, real target: {real_size})")

            # --- Bloc Synthétique ---
            if synth_size == 0:
                logger.warn("Synth split is too small to inject synthethic data, skipping...")
                return
            elif synth_size < len(df_synth_base):
                logger.warn("Synth split is too small to inject all codes from synthethic data, skipping...")
                return
            else:
                rem_synth_needed = synth_size - len(df_synth_base)
                df_synth_chunk = pl.concat([df_synth_base, df_synth_rem.head(rem_synth_needed)])

            # --- Bloc Réel / Original ---
            if real_size == 0:      # Empty df_real_chunk
                df_real_chunk = pl.DataFrame(schema=df_real_base.schema)
            elif real_size < len(df_real_base):
                logger.warn("Synth split is too high but not 1 to inject all codes from real data, skipping...")
                return
            else:
                rem_real_needed = real_size - len(df_real_base)
                df_real_chunk = pl.concat([df_real_base, df_real_rem.head(rem_real_needed)])

            # Validation finale des volumes
            total_extracted = len(df_real_chunk) + len(df_synth_chunk)
            if total_extracted != final_size:
                logger.error(f"Size mismatch for split {split}: got {total_extracted} rows instead of {final_size}")
                return

            # Fusion des deux blocs et brassage final
            df_train = pl.concat([df_real_chunk, df_synth_chunk])
            df_train = df_train.sample(fraction=1.0, shuffle=True, seed=42)
            s3_write(df_train, train_key, fs)

    # Output à destination d'Argo
    print(f"TRAIN_PATHS={','.join(train_paths_emitted)}")


if __name__ == "__main__":
    main()
