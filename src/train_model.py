"""
Step 3 — Model training & evaluation.

Reads preprocessed (train / val / test) parquet files produced by
preprocess_data.py, trains a torchTextClassifiers model, and logs all
metrics + hyper-parameters to MLflow.
"""

import logging
import os
import random
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import numpy as np
import polars as pl
import s3fs
import torch
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer
from torchTextClassifiers.value_encoder import ValueEncoder

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


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def set_seeds():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Main Entrypoint with Hydra
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seeds()
    fs = get_fs()

    # ------------------------------------------------------------------
    # Extraction des chemins passés dynamiquement (via Argo)
    # ------------------------------------------------------------------
    train_path = cfg.get("train_path")
    val_path = cfg.get("val_path")
    test_path = cfg.get("test_path")

    if not all([train_path, val_path, test_path]):
        logger.error("Missing required paths. Ensure train_path, val_path, and test_path are provided.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading data …")
    df_train = s3_read(train_path, fs)
    df_val   = s3_read(val_path,   fs)
    df_test  = s3_read(test_path,  fs)

    logger.info(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    n_classes = df_train["code"].n_unique()
    logger.info(f"Number of classes: {n_classes}")

    # Warn on missing codes
    train_codes = set(df_train["code"])
    missing = (set(df_val["code"]) | set(df_test["code"])) - train_codes
    if missing:
        logger.warning(f"{len(missing)} code(s) missing from training set")
    else:
        logger.info(f"All {len(train_codes)} codes appear in the training set.")

    X_train = df_train["label"].to_numpy()
    y_train = df_train["code"].to_numpy()
    X_val   = df_val["label"].to_numpy()
    y_val   = df_val["code"].to_numpy()
    X_test  = df_test["label"].to_numpy()
    y_test  = df_test["code"].to_numpy()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    encoder = LabelEncoder()
    encoder.fit(y_train)
    value_encoder = ValueEncoder(label_encoder=encoder)

    # ------------------------------------------------------------------
    # Tokenization (instancié depuis le bloc tokenizer de config.yaml)
    # ------------------------------------------------------------------
    logger.info("Training WordPiece tokenizer …")
    tokenizer = WordPieceTokenizer(
        vocab_size=int(cfg.tokenizer.vocab_size),
        output_dim=int(cfg.tokenizer.output_dim),
    )
    tokenizer.train(X_train)

    logger.info(f"Output tensor size : {tokenizer.tokenize(X_train[0]).input_ids.shape}")
    logger.info(f"Vocabulary size    : {tokenizer.vocab_size}")

    # ------------------------------------------------------------------
    # Model & Training Configurations
    # ------------------------------------------------------------------
    model_config = ModelConfig(
        embedding_dim=int(cfg.model.embedding_dim),
        num_classes=n_classes,
    )
    ttc = torchTextClassifiers(
        tokenizer=tokenizer,
        model_config=model_config,
        value_encoder=value_encoder,
    )

    training_config = TrainingConfig(
        num_epochs=int(cfg.training_config.num_epochs),
        batch_size=int(cfg.training_config.batch_size),
        lr=float(cfg.training_config.lr),
        patience_early_stopping=int(cfg.training_config.patience_early_stopping),
        accelerator=cfg.training_config.accelerator,
    )

    # ------------------------------------------------------------------
    # MLflow Setup & Parameters Logging
    # ------------------------------------------------------------------
    mlflow.set_experiment("ttc-injection")
    mlflow.pytorch.autolog()

    # Flat dictionary reconstruit depuis le DictConfig d'Hydra pour MLflow
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    flat_cfg = flatten_dict(cfg_dict)

    with mlflow.start_run():
        mlflow.log_params(flat_cfg)

        # ------------------------------------------------------------------
        # Train
        # ------------------------------------------------------------------
        ttc.train(
            X_train,
            y_train,
            training_config=training_config,
            X_val=X_val,
            y_val=y_val,
            verbose=True,
        )

        # ------------------------------------------------------------------
        # Evaluate
        # ------------------------------------------------------------------
        ttc.pytorch_model.eval()

        # Example predictions (3 random)
        random_indices = random.sample(range(len(X_test)), 3)
        example_texts      = X_test[random_indices]
        example_true_codes = y_test[random_indices]
        top_k_examples = 5
        results_examples = ttc.predict(example_texts, top_k=top_k_examples, explain_with_captum=True)
        for i, text in enumerate(example_texts):
            predicted_codes = [results_examples["prediction"][i][k] for k in range(top_k_examples)]
            confidence      = [results_examples["confidence"][i][k].item() for k in range(top_k_examples)]
            logger.info(f"\nText: {text}")
            logger.info(f"  True code: {example_true_codes[i]}")
            for code, conf in zip(predicted_codes, confidence):
                logger.info(f"  {code}  (confidence: {conf:.3f})")

        # Full test-set predictions
        logger.info("Running predictions on the full test set …")
        results_test = ttc.predict(X_test, top_k=5)

        preds_top5 = np.array(results_test["prediction"])
        conf_top5  = np.array(
            [[c.item() if hasattr(c, "item") else c for c in row]
             for row in results_test["confidence"]]
        )
        y_test_arr = np.array(y_test)

        preds_top1    = preds_top5[:, 0]
        accuracy_top1 = (preds_top1 == y_test_arr).mean()
        accuracy_top3 = np.any(preds_top5[:, :3] == y_test_arr[:, None], axis=1).mean()
        accuracy_top5 = np.any(preds_top5         == y_test_arr[:, None], axis=1).mean()

        threshold          = 0.70
        conf_top1_arr      = conf_top5[:, 0]
        confident_mask     = conf_top1_arr > threshold
        coverage_rate      = confident_mask.mean()
        accuracy_confident = (
            (preds_top1[confident_mask] == y_test_arr[confident_mask]).mean()
            if confident_mask.sum() > 0 else 0.0
        )

        f1_macro    = f1_score(y_test_arr, preds_top1, average="macro",    zero_division=0)
        f1_weighted = f1_score(y_test_arr, preds_top1, average="weighted", zero_division=0)

        logger.info("\n=== GLOBAL PERFORMANCE METRICS ===")
        logger.info(f"Test Accuracy Top-1 : {accuracy_top1:.4f}")
        logger.info(f"Test Accuracy Top-3 : {accuracy_top3:.4f}")
        logger.info(f"Test Accuracy Top-5 : {accuracy_top5:.4f}")
        logger.info(f"\n=== CONFIDENCE ANALYSIS (Threshold > 70%) ===")
        logger.info(f"Coverage Rate        : {coverage_rate:.4f} ({confident_mask.sum()}/{len(y_test_arr)})")
        logger.info(f"Accuracy when conf>70%: {accuracy_confident:.4f}")
        logger.info(f"\n=== IMBALANCE METRICS (747 classes) ===")
        logger.info(f"F1 Macro    : {f1_macro:.4f}")
        logger.info(f"F1 Weighted : {f1_weighted:.4f}")

        mlflow.log_metrics({
            "test_accuracy_top1":      accuracy_top1,
            "test_accuracy_top3":      accuracy_top3,
            "test_accuracy_top5":      accuracy_top5,
            "confidence_coverage_rate": coverage_rate,
            "confidence_accuracy":     accuracy_confident,
            "f1_score_macro":          f1_macro,
            "f1_score_weighted":       f1_weighted,
        })

    logger.info("Done.")


if __name__ == "__main__":
    main()