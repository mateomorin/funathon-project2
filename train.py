"""This script is directly runnable using uv run solutions/1-ttc.py"""

import logging
import random

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import mlflow
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer
from torchTextClassifiers.value_encoder import ValueEncoder

import data

logger = logging.getLogger(__name__)

load_dotenv(override=True)


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@hydra.main(
    version_base=None,
    config_path="",
    config_name="config"
    )
def main(cfg: DictConfig):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Data

    injection_method = cfg["injection"]["method"]
    if injection_method == "fixed_final_size":
        df_train, df_val, df_test = data.fixed_final_size_sampling(cfg)
    elif injection_method == "fixed_original_size":
        df_train, df_val, df_test = data.fixed_original_size_sampling(cfg)
    else:
        raise ValueError(f"{injection_method} is not a valid synthetic injection method.")

    if df_train is None:    # Invalid sampling
        return

    # Preprocessing

    if cfg["tokenizer"]["preprocessed"]:
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords

        french_stopwords = stopwords.words('french')

        df_train = data.preprocess(df_train, text_column="label", stopwords=french_stopwords)
        df_val = data.preprocess(df_val, text_column="label", stopwords=french_stopwords)
        df_test = data.preprocess(df_test, text_column="label", stopwords=french_stopwords)

    n_classes = df_train["code"].n_unique()
    logger.info(f"Number of classes: {n_classes}")

    X_train, y_train = df_train["label"].to_numpy(), df_train["code"].to_numpy()
    X_val, y_val = df_val["label"].to_numpy(), df_val["code"].to_numpy()
    X_test, y_test = df_test["label"].to_numpy(), df_test["code"].to_numpy()

    logger.info(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # Codes

    train_codes = set(df_train["code"])
    val_codes = set(df_val["code"])
    test_codes = set(df_test["code"])
    missing = val_codes.union(test_codes) - train_codes

    if len(missing) > 0:
        logger.warn(f"{len(missing)} code(s) missing from training set")
    else:
        logger.info(f"All {len(train_codes)} codes appear in the training set.")

    encoder = LabelEncoder()
    encoder.fit(y_train)

    value_encoder = ValueEncoder(label_encoder=encoder)

    # Tokenization

    tokenizer = WordPieceTokenizer(vocab_size=cfg["tokenizer"]["vocab_size"], output_dim=cfg["tokenizer"]["output_dim"])
    tokenizer.train(X_train)

    logger.info(f"Output tensor size: {tokenizer.tokenize(X_train[0]).input_ids.shape}")
    logger.info(
        f"Tokens: {tokenizer.tokenizer.convert_ids_to_tokens(tokenizer.tokenize(X_train[0]).input_ids.squeeze(0))}",
    )
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")

    # Model

    embedding_dim = cfg["model"]["embedding_dim"]

    model_config = ModelConfig(
        embedding_dim=embedding_dim,
        num_classes=n_classes,
    )

    ttc = torchTextClassifiers(
        tokenizer=tokenizer,
        model_config=model_config,
        value_encoder=value_encoder,
    )

    mlflow.set_experiment("augmented-codif-ape")

    training_config = TrainingConfig(**cfg["training_config"])

    # Train
    mlflow.pytorch.autolog()

    with mlflow.start_run():
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        flat_cfg = flatten_dict(cfg_dict)
        mlflow.log_params(flat_cfg)

        ttc.train(
            X_train,
            y_train,
            training_config=training_config,
            X_val=X_val,
            y_val=y_val,
            verbose=True,
        )

        # Eval

        ttc.pytorch_model.eval()

        random_indices = random.sample(range(len(X_test)), 3)
        example_texts = X_test[random_indices]
        example_true_codes = y_test[random_indices]
        logger.info(example_texts)
        top_k = 5
        results = ttc.predict(example_texts, top_k=top_k, explain_with_captum=True)
        for i, text in enumerate(example_texts):
            predicted_codes = [results["prediction"][i][k] for k in range(top_k)]
            confidence = [results["confidence"][i][k].item() for k in range(top_k)]
            logger.info(f"\nText: {text}")
            logger.info(f"  True code: {example_true_codes[i]}")
            for code, conf in zip(predicted_codes, confidence):
                logger.info(f"  {code}  (confidence: {conf:.3f})")

        results_test = ttc.predict(X_test, top_k=1)
        preds = results_test["prediction"].squeeze(1)
        accuracy = (preds == y_test).mean()
        logger.info(
            f"Test accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)} correct)"
        )

        logger.info("Logging metrics...")

        mlflow.log_metric("test_accuracy", accuracy)


if __name__ == "__main__":
    main()
