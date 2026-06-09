import logging
import random

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import mlflow
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
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

    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

    mlflow.set_experiment("ttc-injection")

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

        # 1. Affichage de quelques exemples (inchangé mais isolé)
        random_indices = random.sample(range(len(X_test)), 3)
        example_texts = X_test[random_indices]
        example_true_codes = y_test[random_indices]
        logger.info(example_texts)
        top_k_examples = 5

        results_examples = ttc.predict(example_texts, top_k=top_k_examples, explain_with_captum=True)
        for i, text in enumerate(example_texts):
            predicted_codes = [results_examples["prediction"][i][k] for k in range(top_k_examples)]
            confidence = [results_examples["confidence"][i][k].item() for k in range(top_k_examples)]
            logger.info(f"\nText: {text}")
            logger.info(f"  True code: {example_true_codes[i]}")
            for code, conf in zip(predicted_codes, confidence):
                logger.info(f"  {code}  (confidence: {conf:.3f})")

        # 2. Prédiction globale sur le set de test pour le calcul des métriques (Top-5)
        logger.info("Running predictions on the full test set...")
        results_test = ttc.predict(X_test, top_k=5)  # On demande le top 5 d'un coup

        # Extraction des prédictions et confiances sous forme de arrays numpy pour manipuler facilement
        preds_top5 = np.array(results_test["prediction"])   # Forme attendue : (nb_exemples, 5)
        conf_top5 = np.array([[c.item() if hasattr(c, 'item') else c for c in row] for row in results_test["confidence"]]) # Forme : (nb_exemples, 5)
        y_test_arr = np.array(y_test)

        # --- CALCUL DES TOP-K ACCURACY ---
        # Top-1
        preds_top1 = preds_top5[:, 0]
        accuracy_top1 = (preds_top1 == y_test_arr).mean()

        # Top-3
        correct_top3 = np.any(preds_top5[:, :3] == y_test_arr[:, None], axis=1)
        accuracy_top3 = correct_top3.mean()

        # Top-5
        correct_top5 = np.any(preds_top5 == y_test_arr[:, None], axis=1)
        accuracy_top5 = correct_top5.mean()

        # --- ANALYSE DE LA CONFIANCE (Seuil > 70%) ---
        threshold = 0.70
        conf_top1 = conf_top5[:, 0]  # Confiance accordée au premier choix

        # Taux de couverture : % de cas où le modèle dépasse le seuil
        confident_mask = conf_top1 > threshold
        coverage_rate = confident_mask.mean()

        # Accuracy filtrée : Précision du modèle uniquement lorsqu'il est confiant
        if confident_mask.sum() > 0:
            accuracy_confident = (preds_top1[confident_mask] == y_test_arr[confident_mask]).mean()
        else:
            accuracy_confident = 0.0

        # --- F1-Scores ---
        f1_macro = f1_score(y_test_arr, preds_top1, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test_arr, preds_top1, average='weighted', zero_division=0)

        # --- LOGGING DES RÉSULTATS ---
        logger.info("\n=== GLOBAL PERFORMANCE METRICS ===")
        logger.info(f"Test Accuracy Top-1 : {accuracy_top1:.4f}")
        logger.info(f"Test Accuracy Top-3 : {accuracy_top3:.4f}")
        logger.info(f"Test Accuracy Top-5 : {accuracy_top5:.4f}")

        logger.info("\n=== CONFIDENCE ANALYSIS (Seuil > 70%) ===")
        logger.info(f"Coverage Rate (Conf > 70%)   : {coverage_rate:.4f} ({confident_mask.sum()}/{len(y_test_arr)})")
        logger.info(f"Accuracy when Conf > 70%     : {accuracy_confident:.4f}")

        logger.info("\n=== IMBALANCE METRICS (747 Classes) ===")
        logger.info(f"F1-Score (Macro)    : {f1_macro:.4f}  <- (Indicateur de performance sur les classes rares)")
        logger.info(f"F1-Score (Weighted) : {f1_weighted:.4f}")

        # --- MLFLOW LOGGING ---
        logger.info("Logging metrics to MLflow...")
        mlflow.log_metrics({
            "test_accuracy_top1": accuracy_top1,
            "test_accuracy_top3": accuracy_top3,
            "test_accuracy_top5": accuracy_top5,
            "confidence_coverage_rate": coverage_rate,
            "confidence_accuracy": accuracy_confident,
            "f1_score_macro": f1_macro,
            "f1_score_weighted": f1_weighted
        })

    return df_train, df_val, df_test, ttc


if __name__ == "__main__":
    main()
