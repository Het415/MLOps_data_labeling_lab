import hashlib
import importlib
import json
import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_union

DATA_PATH = "data/sms_spam.csv"
METRICS_PATH = "results/metrics.json"
SEED = 123


def download_nlp_resources():
    """Fetch the NLTK corpora and spaCy model the notebooks need, if missing."""
    import nltk
    import spacy

    for pkg in ["wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)
    if not spacy.util.is_package("en_core_web_sm"):
        spacy.cli.download("en_core_web_sm")
        importlib.invalidate_caches()


def load_spam_dataset(load_train_labels=False, split_dev_valid=False, dev_size=300):
    """Load the SMS data and split it into train, dev, valid and test.

    Exact duplicate messages are dropped first so the same text can't land
    in both train and test. All splits are stratified on the label.
    Dev is a small labeled sample drawn from train, like in the original lab.
    """
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates("text").reset_index(drop=True)

    df_rest, df_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df.label
    )
    df_train, df_valid = train_test_split(
        df_rest, test_size=0.125, random_state=SEED, stratify=df_rest.label
    )
    _, df_dev = train_test_split(
        df_train, test_size=dev_size, random_state=SEED, stratify=df_train.label
    )

    df_train = df_train.reset_index(drop=True)
    df_dev = df_dev.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    if not load_train_labels:
        df_train["label"] = -1

    if split_dev_valid:
        return df_train, df_dev, df_valid, df_test
    return df_train, df_test


def make_tfidf():
    """Word 1-2 grams plus character 2-5 grams, both TF-IDF weighted."""
    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
    )


def binary_scores(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def save_metrics(section, values, path=METRICS_PATH):
    """Write one notebook's numbers under its own key in metrics.json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metrics = {}
    if os.path.exists(path):
        with open(path) as f:
            metrics = json.load(f)
    metrics[section] = values
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def preview_tfs(df, tfs):
    """Show the first successful output of each TF on a shuffled copy of df."""
    rows = []
    for f in tfs:
        for _, row in df.sample(frac=1, random_state=2).iterrows():
            out = f(row)
            if out is not None:
                rows.append(
                    OrderedDict(
                        {"TF Name": f.name, "Original Text": row.text, "Transformed Text": out.text}
                    )
                )
                break
    return pd.DataFrame(rows)


def hash_tokens(text, max_length=30, num_buckets=30000):
    """Lowercase, split, md5-hash each token into a bucket, then pad or truncate."""
    tokens = re.sub(r"[!\"#$%&()*+,\-./:;<=>?@\[\\\]^_`{|}~\t\n]", " ", text.lower()).split()
    ids = [int(hashlib.md5(t.encode()).hexdigest(), 16) % (num_buckets - 1) + 1 for t in tokens]
    return ids[:max_length] + [0] * (max_length - len(ids))


def featurize_df_tokens(df):
    return np.array([hash_tokens(t) for t in df.text])


def get_keras_lstm(num_buckets=30000, embed_dim=16, rnn_state_size=64, optimizer="adagrad"):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(30,)),
            tf.keras.layers.Embedding(num_buckets, embed_dim),
            tf.keras.layers.LSTM(rnn_state_size, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer, "binary_crossentropy", metrics=["accuracy"])
    return model


def df_to_features(vectorizer, df, split):
    """Bag-of-words features as a dense array, fitting the vectorizer on train only."""
    if split == "train":
        feats = vectorizer.fit_transform(df.text.tolist())
    else:
        feats = vectorizer.transform(df.text.tolist())
    return np.asarray(feats.todense()), df["label"].values


def create_dict_dataloader(X, Y, split, **kwargs):
    import torch
    from snorkel.classification.data import DictDataLoader, DictDataset

    ds = DictDataset.from_tensors(torch.FloatTensor(X), torch.LongTensor(Y), split)
    return DictDataLoader(ds, **kwargs)


def get_pytorch_mlp(hidden_dim, num_layers):
    import torch.nn as nn

    layers = []
    for _ in range(num_layers):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    return nn.Sequential(*layers)
