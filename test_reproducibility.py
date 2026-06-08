import pytest
import torch
import random
import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning
from torchTextClassifiers import torchTextClassifiers, ModelConfig
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers.value_encoder import ValueEncoder
from torchTextClassifiers.tokenizers import WordPieceTokenizer

# Import de vos fonctions locales
import data
import train


@pytest.fixture
def base_config():
    """Génère la configuration de test calquée sur votre structure YAML/Hydra."""
    return OmegaConf.create({
        "data": {
            "synth_path": "s3://projet-ape/synthetic_data_test/naive/NAF2025_FR/retext-2026-05-29_gemma4-26b-moe_temp10_fewshot6_exhaustive.parquet",
            "original_train_path": "s3://projet-ape/data/08112022_27102024/naf2025/split/df_train.parquet",
            "original_val_path": "s3://projet-ape/data/08112022_27102024/naf2025/split/df_val.parquet",
            "original_test_path": "s3://projet-ape/data/08112022_27102024/naf2025/split/df_test.parquet"
        },
        "injection": {
            "method": "fixed_final_size",
            "synth_split": 0.3,
            "final_size": 15000,         # Taille réduite pour un test rapide
            "subtrain_size": 10000,
            "val_test_sample": 0.01
        },
        "tokenizer": {
            "preprocessed": True,     # Désactivé pour accélérer le test unitaire
            "vocab_size": 1000,
            "output_dim": 32
        },
        "model": {
            "embedding_dim": 64
        },
        "training_config": {
            "num_epochs": 1,               # Une seule epoch demandée
            "batch_size": 16,
            "lr": 1e-3,
            "patience_early_stopping": 2,
            "accelerator": "cpu"
        }
    })


def set_all_seeds():
    """Fixe l'intégralité des graines aléatoires pour une reproductibilité stricte."""
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    pytorch_lightning.seed_everything(42)


# --- FONCTIONS CONNEXES POUR LE SAMPLING ---

def run_fixed_final_size_sampling(cfg):
    """Exécute et retourne le sampling fixed_final_size."""
    set_all_seeds()
    return data.fixed_final_size_sampling(cfg)


def run_fixed_original_size_sampling(cfg):
    """Exécute et retourne le sampling fixed_original_size."""
    set_all_seeds()
    return data.fixed_original_size_sampling(cfg)


# --- TESTS UNITAIRES ---

def test_samplings_reproducibility(base_config):
    """
    Vérifie de manière isolée que les fonctions de sampling de data.py
    sont strictement reproductibles.
    """
    # Test pour fixed_final_size
    df_train1, df_val1, df_test1 = run_fixed_final_size_sampling(base_config)
    df_train2, df_val2, df_test2 = run_fixed_final_size_sampling(base_config)

    assert df_train1.equals(df_train2), "Le dataset d'entraînement a divergé (fixed_final_size) !"
    assert df_val1.equals(df_val2), "Le dataset de validation a divergé (fixed_final_size) !"
    assert df_test1.equals(df_test2), "Le dataset de test a divergé (fixed_final_size) !"

    # Test pour fixed_original_size
    cfg_orig = base_config.copy()
    cfg_orig["injection"]["method"] = "fixed_original_size"

    df_train_o1, df_val_o1, df_test_o1 = run_fixed_original_size_sampling(cfg_orig)
    df_train_o2, df_val_o2, df_test_o2 = run_fixed_original_size_sampling(cfg_orig)

    assert df_train_o1.equals(df_train_o2), "Le dataset d'entraînement a divergé (fixed_original_size) !"


def init_ttc(base_config):
    # Data

    injection_method = base_config["injection"]["method"]
    if injection_method == "fixed_final_size":
        df_train, df_val, df_test = data.fixed_final_size_sampling(base_config)
    elif injection_method == "fixed_original_size":
        df_train, df_val, df_test = data.fixed_original_size_sampling(base_config)
    else:
        raise ValueError(f"{injection_method} is not a valid synthetic injection method.")

    if df_train is None:    # Invalid sampling
        return

    # Preprocessing

    if base_config["tokenizer"]["preprocessed"]:
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords

        french_stopwords = stopwords.words('french')

        df_train = data.preprocess(df_train, text_column="label", stopwords=french_stopwords)
        df_val = data.preprocess(df_val, text_column="label", stopwords=french_stopwords)
        df_test = data.preprocess(df_test, text_column="label", stopwords=french_stopwords)

    n_classes = df_train["code"].n_unique()

    X_train, y_train = df_train["label"].to_numpy(), df_train["code"].to_numpy()

    # Codes

    encoder = LabelEncoder()
    encoder.fit(y_train)

    value_encoder = ValueEncoder(label_encoder=encoder)

    # Tokenization

    tokenizer = WordPieceTokenizer(vocab_size=base_config["tokenizer"]["vocab_size"], output_dim=base_config["tokenizer"]["output_dim"])
    tokenizer.train(X_train)

    # Model

    embedding_dim = base_config["model"]["embedding_dim"]

    model_config = ModelConfig(
        embedding_dim=embedding_dim,
        num_classes=n_classes,
    )

    ttc = torchTextClassifiers(
        tokenizer=tokenizer,
        model_config=model_config,
        value_encoder=value_encoder,
    )

    return ttc


def test_model_initialization_reproducibility(base_config):
    """
    Vérifie que l'initialisation brute du modèle (poids et plongements) 
    est strictement identique entre deux exécutions distinctes, 
    avant toute phase d'entraînement (backpropagation).
    """

    set_all_seeds()
    ttc1 = init_ttc(base_config)

    set_all_seeds()
    ttc2 = init_ttc(base_config)

    state_dict_1 = ttc1.pytorch_model.state_dict()
    state_dict_2 = ttc2.pytorch_model.state_dict()

    for key in state_dict_1.keys():
        assert torch.equal(state_dict_1[key], state_dict_2[key]), f"Divergence dans la couche : {key}"


# def test_sampling_and_training_reproducibility(base_config):
#     """
#     Vérifie la reproductibilité globale en appelant directement train.main().
#     Valide les DataFrames retournés ainsi que l'état final du modèle.
#     """

#     # --- PASSE 1 ---
#     set_all_seeds()
#     df_train1, df_val1, df_test1, ttc1 = train.main.__wrapped__(base_config)
#     state_dict_1 = ttc1.pytorch_model.state_dict()

#     # --- PASSE 2 ---
#     set_all_seeds()
#     df_train2, df_val2, df_test2, ttc2 = train.main.__wrapped__(base_config)
#     state_dict_2 = ttc2.pytorch_model.state_dict()

#     # 1. Validation des outputs de données du main()
#     assert df_train1.equals(df_train2), "df_train a divergé entre les deux exécutions de main() !"
#     assert df_val1.equals(df_val2), "df_val a divergé entre les deux exécutions de main() !"
#     assert df_test1.equals(df_test2), "df_test a divergé entre les deux exécutions de main() !"

#     # 2. Validation des poids du modèle TTC issu de main()
#     for key in state_dict_1.keys():
#         # atol=1e-4 autorise une différence absolue de 0.0001 maximum entre les poids
#         assert torch.allclose(state_dict_1[key], state_dict_2[key], atol=1e-4), \
#             f"Divergence significative dans la couche : {key}"
