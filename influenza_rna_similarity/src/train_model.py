import itertools
import numpy as np
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from load_data import load_fasta_files
from preprocess_data import preprocess_dataframe


def build_kmer_vocab(k=3):
    bases = ["A", "C", "G", "U"]
    return ["".join(p) for p in itertools.product(bases, repeat=k)]


def kmer_freq_vector(kmers, vocab):
    counts = Counter(kmers)
    return np.array([counts.get(k, 0) for k in vocab])


def compute_segment_similarity(df, vocab):
    results = {}

    for segment in df["segment"].unique():
        sub = df[df["segment"] == segment]

        X = np.vstack(
            sub["kmers"].apply(lambda x: kmer_freq_vector(x, vocab))
        )

        X = normalize(X, norm="l2")
        sim_matrix = cosine_similarity(X)

        results[segment] = {
            "ids": sub["id"].values,
            "similarity": sim_matrix
        }

    return results


def compute_user_similarity(user_kmers, df, vocab, segment):
    sub = df[df["segment"] == segment]

    X_db = np.vstack(
        sub["kmers"].apply(lambda x: kmer_freq_vector(x, vocab))
    )
    X_db = normalize(X_db, norm="l2")

    user_vec = kmer_freq_vector(user_kmers, vocab).reshape(1, -1)
    user_vec = normalize(user_vec, norm="l2")

    scores = cosine_similarity(user_vec, X_db)[0]

    return list(zip(sub["id"].values, scores))


if __name__ == "__main__":

    segment_files = {
        "PB2": r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\PB2.fa",
        "PB1": r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\PB1.fa",
        "PA":  r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\PA.fa",
        "HA":  r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\HA.fa",
        "NP":  r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\NP.fa",
        "NA":  r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\NA.fa",
        "M":   r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\M.fa",
        "NS":  r"C:\Users\rjpms\OneDrive\Documents\ml_dl\ai-ml-engineering\00.influenza_rna_research\data\NS.fa"
    }

    df = load_fasta_files(segment_files)
    df = preprocess_dataframe(df, k=3)
    vocab = build_kmer_vocab(k=3)

    similarity_results = compute_segment_similarity(df, vocab)

    print("Cosine similarity computed for segments:")
    print(list(similarity_results.keys()))
