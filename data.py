import logging

from dotenv import load_dotenv
import polars as pl
import s3fs

logger = logging.getLogger(__name__)

load_dotenv(override=True)


def fetch_original_data(path):
    df = pl.read_parquet(path)
    df = df.rename(mapping={"nace2025": "code", "libelle": "label"})[["code", "label"]]
    df = df.with_columns(
        (pl.col("code").str.slice(0, 2) + "." + pl.col("code").str.slice(2)).alias("code")
    )
    return df


def sample_with_all_codes(df, code_column, sample_size):
    df_guaranteed = df.unique(subset=[code_column], keep="first", maintain_order=True)
    df_remaining = df.join(df_guaranteed, on=df.columns, how="anti", maintain_order="left_right")
    remaining_size = sample_size - len(df_guaranteed)
    if remaining_size <= 0:
        logger.warn(f"The sampling size {sample_size} is smaller than the number of unique codes {len(df_guaranteed)}.")
        return df_guaranteed
    df_remaining_sampled = df_remaining.sample(n=remaining_size, seed=42)
    return pl.concat([df_guaranteed, df_remaining_sampled]).sample(fraction=1.0, seed=42)


def fixed_original_size_sampling(cfg):
    fs = s3fs.S3FileSystem(
        endpoint_url="https://minio.lab.sspcloud.fr",
        client_kwargs={"region_name": "us-east-1"},
    )

    synth_split = cfg["injection"]["synth_split"]
    assert 0 <= synth_split <= 1

    # Load data

    with fs.open(cfg["data"]["synth_path"]) as f:
        df_synth = pl.read_parquet(f)
        df_synth.drop_in_place("name")

    with fs.open(cfg["data"]["original_val_path"]) as f:
        df_val = fetch_original_data(f)

    with fs.open(cfg["data"]["original_test_path"]) as f:
        df_test = fetch_original_data(f)

    if synth_split == 1:
        df_train = df_synth
    else:
        with fs.open(cfg["data"]["original_train_path"]) as f:      # Sampling only a proportion
            df_train = fetch_original_data(f)
            df_train = sample_with_all_codes(
                df=df_train,
                code_column="code",
                sample_size=cfg["injection"]["train_original_sample"]
            )

        if synth_split > 0:
            synth_size = round(synth_split * len(df_train) / (1 - synth_split))

            if synth_size > len(df_synth) * 1.1:
                logger.warn(f"synth_split is too high to sample enough labels: {synth_size} synth labels wanted vs {len(df_synth)} synth labels.")
                return None, None, None

            subdf_synth = df_synth.sample(min(len(df_synth), synth_size), seed=42, shuffle=True)
            df_train = pl.concat([df_train, subdf_synth])
            df_train = df_train.sample(fraction=1.0, seed=42, shuffle=True)

    df_val = df_val.sample(n=cfg["injection"]["val_test_sample"], shuffle=True, seed=42)
    df_test = df_test.sample(n=cfg["injection"]["val_test_sample"], shuffle=True, seed=42)

    return df_train, df_val, df_test


def fixed_final_size_sampling(cfg):
    fs = s3fs.S3FileSystem(
        endpoint_url="https://minio.lab.sspcloud.fr",
        client_kwargs={"region_name": "us-east-1"},
    )

    synth_split = cfg["injection"]["synth_split"]
    assert 0 <= synth_split <= 1
    final_size = cfg["injection"]["final_size"]
    synth_size = round(synth_split * final_size)
    real_size = round((1 - synth_split) * final_size)

    # Load data

    with fs.open(cfg["data"]["original_val_path"]) as f:
        df_val = fetch_original_data(f)

    with fs.open(cfg["data"]["original_test_path"]) as f:
        df_test = fetch_original_data(f)

    if synth_split == 1:
        with fs.open(cfg["data"]["synth_path"]) as f:
            df_train = pl.read_parquet(f)
            df_train.drop_in_place("name")
            if final_size > len(df_train):
                logger.warn(f"Size of the synthetic dataset ({len(df_train)}) is too low for the expected size {final_size}.")
                return None, None, None
            df_train = sample_with_all_codes(
                df=df_train,
                code_column="code",
                sample_size=final_size
            )
    elif synth_split == 0:
        with fs.open(cfg["data"]["original_train_path"]) as f:
            df_train = fetch_original_data(f)
            if final_size > len(df_train):
                logger.warn(f"Size of the original dataset ({len(df_train)}) is too low for the expected size {final_size}.")
                return None, None, None
            df_train = sample_with_all_codes(
                df=df_train,
                code_column="code",
                sample_size=final_size
            )
    else:
        with fs.open(cfg["data"]["synth_path"]) as f:
            df_synth = pl.read_parquet(f)
            if synth_size > len(df_synth):
                logger.warn(f"Size of the synthetic dataset ({len(df_synth)}) is too low for the expected size {synth_size}.")
                return None, None, None
            df_synth = sample_with_all_codes(
                df=df_synth,
                code_column="code",
                sample_size=synth_size
            )
            df_synth.drop_in_place("name")
        with fs.open(cfg["data"]["original_train_path"]) as f:
            df_real = fetch_original_data(f)
            if real_size > len(df_real):
                logger.warn(f"Size of the original dataset ({len(df_real)}) is too low for the expected size {real_size}.")
                return None, None, None
            df_real = sample_with_all_codes(
                df=df_real,
                code_column="code",
                sample_size=real_size
            )

        df_train = pl.concat([df_real, df_synth])
        df_train = df_train.sample(fraction=1.0, shuffle=True, seed=42)

    df_val = df_val.sample(n=cfg["injection"]["val_test_sample"], shuffle=True, seed=42)
    df_test = df_test.sample(n=cfg["injection"]["val_test_sample"], shuffle=True, seed=42)

    return df_train, df_val, df_test


def preprocess(df: pl.DataFrame, text_column: str, stopwords: list) -> pl.DataFrame:
    """
    Nettoie et normalise une colonne textuelle d'un DataFrame Polars
    pour préparer une codification automatique.

    Retourne un nouveau DataFrame (Polars ne gère pas le 'inplace').
    """
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
