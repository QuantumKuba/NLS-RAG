"""
Experiment 6: OCR Noise Injection & Retrieval Robustness.

Progressively injects synthetic OCR noise into high-quality documents and
measures retrieval degradation. Compares BM25 (sparse) vs dense retrieval
robustness to demonstrate that the dataset supports noise-resilient IR research.

Usage:
    python experiments/experiment6_noise_robustness.py --corpus data/corpus.jsonl --output results/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Prevent tokenizer parallelism segfaults on ARM Mac / Python 3.9
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_corpus(corpus_path: Path) -> list[dict]:
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


# ═══════════════════════════════════════════════════════
# Noise Injection
# ═══════════════════════════════════════════════════════

# Common OCR confusion pairs based on real OCR errors
OCR_CONFUSIONS = {
    'a': ['o', 'e', 'u'],
    'b': ['h', 'd', 'p'],
    'c': ['e', 'o'],
    'd': ['b', 'cl'],
    'e': ['c', 'o', 'a'],
    'f': ['t', 's'],
    'g': ['q', '9'],
    'h': ['b', 'n', 'k'],
    'i': ['l', '1', 'j', '!'],
    'j': ['i', 'l'],
    'k': ['h', 'x'],
    'l': ['i', '1', '|'],
    'm': ['rn', 'nn'],
    'n': ['m', 'u', 'ri'],
    'o': ['0', 'c', 'a'],
    'p': ['b', 'q'],
    'q': ['g', 'p'],
    'r': ['n', 'v'],
    's': ['5', 'f'],
    't': ['f', 'l'],
    'u': ['n', 'v', 'a'],
    'v': ['u', 'r', 'y'],
    'w': ['vv', 'uv'],
    'x': ['k', 'z'],
    'y': ['v', 'g'],
    'z': ['x', '2'],
}

# Types of noise
NOISE_TYPES = {
    "substitution": 0.50,  # Swap with OCR-confused char
    "deletion": 0.20,       # Delete character
    "insertion": 0.15,      # Insert random char
    "transposition": 0.15,  # Swap adjacent chars
}


def inject_noise(text: str, error_rate: float, seed: int = 42) -> tuple[str, float]:
    """Inject synthetic OCR noise at specified character error rate.
    
    Returns: (noisy_text, actual_cer)
    """
    rng = random.Random(seed)
    chars = list(text)
    n_errors = int(len(chars) * error_rate)

    # Select random positions to corrupt
    positions = rng.sample(range(len(chars)), min(n_errors, len(chars)))
    actual_errors = 0

    for pos in sorted(positions, reverse=True):  # Reverse to handle insertions
        noise_type = rng.choices(
            list(NOISE_TYPES.keys()),
            list(NOISE_TYPES.values()),
        )[0]

        if noise_type == "substitution":
            c = chars[pos].lower()
            if c in OCR_CONFUSIONS:
                replacement = rng.choice(OCR_CONFUSIONS[c])
                if chars[pos].isupper():
                    replacement = replacement.upper()
                chars[pos] = replacement
                actual_errors += 1
            else:
                # Random substitution
                chars[pos] = rng.choice("abcdefghijklmnopqrstuvwxyz")
                actual_errors += 1

        elif noise_type == "deletion":
            if len(chars) > 10:  # Don't delete too much
                chars.pop(pos)
                actual_errors += 1

        elif noise_type == "insertion":
            chars.insert(pos, rng.choice("abcdefghijklmnopqrstuvwxyz "))
            actual_errors += 1

        elif noise_type == "transposition":
            if pos < len(chars) - 1:
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                actual_errors += 1

    noisy_text = "".join(chars)
    actual_cer = actual_errors / max(len(text), 1)
    return noisy_text, actual_cer


# ═══════════════════════════════════════════════════════
# BM25 Retrieval
# ═══════════════════════════════════════════════════════

def bm25_score(query_tokens: list[str], doc_tokens: list[str],
               doc_freqs: dict, n_docs: int, avg_dl: float,
               k1: float = 1.2, b: float = 0.75) -> float:
    score = 0.0
    dl = len(doc_tokens)
    tf_doc = Counter(doc_tokens)
    for term in query_tokens:
        if term not in tf_doc:
            continue
        tf = tf_doc[term]
        df = doc_freqs.get(term, 0)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        score += idf * tf_norm
    return score


def run_bm25_retrieval(corpus_texts: list[str], query_texts: list[str],
                       doc_ids: list[str], k: int = 10) -> dict:
    """Run BM25 and return per-query rankings."""
    doc_tokens_all = [tokenize(t) for t in corpus_texts]
    doc_freqs = Counter()
    for tokens in doc_tokens_all:
        for t in set(tokens):
            doc_freqs[t] += 1
    n_docs = len(corpus_texts)
    avg_dl = sum(len(t) for t in doc_tokens_all) / n_docs if n_docs > 0 else 1

    results = {}
    for q_text in query_texts:
        q_tokens = tokenize(q_text)
        scores = []
        for i, doc_tokens in enumerate(doc_tokens_all):
            s = bm25_score(q_tokens, doc_tokens, doc_freqs, n_docs, avg_dl)
            scores.append((doc_ids[i], s))
        scores.sort(key=lambda x: x[1], reverse=True)
        results[q_text] = {did: s for did, s in scores[:k]}

    return results


# ═══════════════════════════════════════════════════════
# Dense Retrieval
# ═══════════════════════════════════════════════════════

def run_dense_retrieval(corpus_texts: list[str], query_texts: list[str],
                        doc_ids: list[str], model, k: int = 10) -> dict:
    """Run dense retrieval using pre-loaded model."""
    import faiss
    import gc

    doc_embeddings = model.encode(
        [t[:8192] for t in corpus_texts],
        batch_size=32, show_progress_bar=False,
        normalize_embeddings=True,
    )

    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings.astype(np.float32))

    del doc_embeddings
    gc.collect()

    q_embeddings = model.encode(query_texts, normalize_embeddings=True)
    scores, indices = index.search(q_embeddings.astype(np.float32), k)

    results = {}
    for i, q_text in enumerate(query_texts):
        results[q_text] = {}
        for j in range(k):
            if indices[i][j] >= 0:
                results[q_text][doc_ids[indices[i][j]]] = float(scores[i][j])

    gc.collect()
    return results


# ═══════════════════════════════════════════════════════
# Noise Robustness Analysis
# ═══════════════════════════════════════════════════════

def compute_ranking_overlap(original_results: dict, noisy_results: dict, k: int = 10) -> float:
    """Compute mean overlap between original and noisy top-k rankings."""
    overlaps = []
    for q_text in original_results:
        if q_text not in noisy_results:
            continue
        orig_top = set(list(original_results[q_text].keys())[:k])
        noisy_top = set(list(noisy_results[q_text].keys())[:k])
        if orig_top:
            overlap = len(orig_top & noisy_top) / len(orig_top)
            overlaps.append(overlap)
    return sum(overlaps) / len(overlaps) if overlaps else 0


def compute_score_degradation(original_results: dict, noisy_results: dict) -> float:
    """Compute mean score degradation (relative drop in top-1 score)."""
    degradations = []
    for q_text in original_results:
        if q_text not in noisy_results:
            continue
        orig_scores = list(original_results[q_text].values())
        noisy_scores = list(noisy_results[q_text].values())
        if orig_scores and noisy_scores:
            orig_top = orig_scores[0]
            noisy_top = noisy_scores[0]
            if orig_top > 0:
                degradations.append((orig_top - noisy_top) / orig_top)
    return sum(degradations) / len(degradations) if degradations else 0


def run_robustness_experiment(docs: list[dict], error_rates: list[float],
                               model_name: str = "BAAI/bge-small-en-v1.5") -> dict:
    """Run full noise robustness experiment."""
    from sentence_transformers import SentenceTransformer
    import gc

    # Select high-quality documents
    high_docs = [d for d in docs if d.get("ocr_quality_tier") == "high"]
    if len(high_docs) < 10:
        high_docs = sorted(docs, key=lambda d: d.get("ocr_quality", 0), reverse=True)[:50]

    print(f"  Using {len(high_docs)} high-quality documents")

    doc_ids = [d["doc_id"] for d in high_docs]
    original_texts = [d["text"] for d in high_docs]

    # Test queries (representative historical queries)
    test_queries = [
        "colonial administration correspondence",
        "trade routes India East Indies",
        "missionary expedition Africa",
        "railway construction British Empire",
        "manuscript letters diplomatic",
        "botanical specimens natural history",
        "military campaign strategic",
        "legislative governance council",
    ]

    # Load model once
    print(f"  Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Baseline (no noise)
    print("  Running baseline (0% noise)...")
    bm25_baseline = run_bm25_retrieval(original_texts, test_queries, doc_ids)
    dense_baseline = run_dense_retrieval(original_texts, test_queries, doc_ids, model)

    results = {
        "n_documents": len(high_docs),
        "n_queries": len(test_queries),
        "error_rates": [],
    }

    for cer in error_rates:
        print(f"\n  Running with CER = {cer:.0%}...")

        # Inject noise
        noisy_texts = []
        actual_cers = []
        for i, text in enumerate(original_texts):
            noisy, actual = inject_noise(text, cer, seed=42 + i)
            noisy_texts.append(noisy)
            actual_cers.append(actual)

        mean_actual_cer = sum(actual_cers) / len(actual_cers)
        print(f"    Mean actual CER: {mean_actual_cer:.4f}")

        # BM25 on noisy corpus
        bm25_noisy = run_bm25_retrieval(noisy_texts, test_queries, doc_ids)
        bm25_overlap = compute_ranking_overlap(bm25_baseline, bm25_noisy)
        bm25_degradation = compute_score_degradation(bm25_baseline, bm25_noisy)

        # Dense on noisy corpus
        dense_noisy = run_dense_retrieval(noisy_texts, test_queries, doc_ids, model)
        dense_overlap = compute_ranking_overlap(dense_baseline, dense_noisy)
        dense_degradation = compute_score_degradation(dense_baseline, dense_noisy)

        print(f"    BM25:  overlap={bm25_overlap:.4f}, degradation={bm25_degradation:.4f}")
        print(f"    Dense: overlap={dense_overlap:.4f}, degradation={dense_degradation:.4f}")

        results["error_rates"].append({
            "target_cer": cer,
            "actual_cer": round(mean_actual_cer, 4),
            "bm25": {
                "ranking_overlap_at10": round(bm25_overlap, 4),
                "score_degradation": round(bm25_degradation, 4),
            },
            "dense": {
                "ranking_overlap_at10": round(dense_overlap, 4),
                "score_degradation": round(dense_degradation, 4),
            },
        })

        gc.collect()

    del model
    gc.collect()

    return results


# ═══════════════════════════════════════════════════════
# Figure Generation
# ═══════════════════════════════════════════════════════

def generate_figures(results: dict, output_dir: Path):
    """Generate publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not available, skipping figures")
        return

    fig_dir = output_dir.parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
    })

    rates = results["error_rates"]
    if not rates:
        return

    # Add baseline (0% noise) point
    cers = [0] + [r["target_cer"] for r in rates]
    bm25_overlaps = [1.0] + [r["bm25"]["ranking_overlap_at10"] for r in rates]
    dense_overlaps = [1.0] + [r["dense"]["ranking_overlap_at10"] for r in rates]
    bm25_degs = [0.0] + [r["bm25"]["score_degradation"] for r in rates]
    dense_degs = [0.0] + [r["dense"]["score_degradation"] for r in rates]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Ranking Overlap
    axes[0].plot(cers, bm25_overlaps, "o-", color="#C44E52", linewidth=2,
                 markersize=8, label="BM25 (Sparse)")
    axes[0].plot(cers, dense_overlaps, "s-", color="#4C72B0", linewidth=2,
                 markersize=8, label="Dense (BGE)")
    axes[0].set_xlabel("Character Error Rate (CER)")
    axes[0].set_ylabel("Ranking Overlap@10")
    axes[0].set_title("(a) Retrieval Stability Under OCR Noise")
    axes[0].legend()
    axes[0].set_xlim(-0.01, max(cers) + 0.02)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(cers)
    axes[0].set_xticklabels([f"{c:.0%}" for c in cers])

    # (b) Score Degradation
    axes[1].plot(cers, bm25_degs, "o-", color="#C44E52", linewidth=2,
                 markersize=8, label="BM25 (Sparse)")
    axes[1].plot(cers, dense_degs, "s-", color="#4C72B0", linewidth=2,
                 markersize=8, label="Dense (BGE)")
    axes[1].set_xlabel("Character Error Rate (CER)")
    axes[1].set_ylabel("Relative Score Degradation")
    axes[1].set_title("(b) Score Degradation Under OCR Noise")
    axes[1].legend()
    axes[1].set_xlim(-0.01, max(cers) + 0.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(cers)
    axes[1].set_xticklabels([f"{c:.0%}" for c in cers])

    plt.tight_layout()
    plt.savefig(fig_dir / "fig13_noise_robustness.png", bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved fig13_noise_robustness.png")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="OCR Noise Injection & Robustness")
    parser.add_argument("--corpus", type=str, default="data/corpus.jsonl")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading corpus...")
    docs = load_corpus(corpus_path)
    print(f"  {len(docs)} documents loaded")

    # Error rates to test
    error_rates = [0.05, 0.10, 0.20, 0.30]

    print(f"\nRunning noise robustness experiment...")
    print(f"  Error rates: {[f'{r:.0%}' for r in error_rates]}")

    results = run_robustness_experiment(docs, error_rates)

    output_file = output_dir / "experiment6_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")

    print("\nGenerating figures...")
    generate_figures(results, output_dir)

    print("\n" + "=" * 60)
    print("Experiment 6 Complete: OCR Noise Injection & Robustness")
    print("=" * 60)


if __name__ == "__main__":
    main()
