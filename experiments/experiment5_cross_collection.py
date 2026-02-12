"""
Experiment 5: Cross-Collection Comparative Analysis.

Compares the two NLS sub-collections (indiaraj vs africa_and_new_imperialism)
across OCR quality, retrieval effectiveness, vocabulary overlap, and
cross-collection transfer retrieval.

Usage:
    python experiments/experiment5_cross_collection.py --corpus data/corpus.jsonl --output results/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
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
# Vocabulary Overlap
# ═══════════════════════════════════════════════════════

def compute_vocabulary_overlap(docs: list[dict]) -> dict:
    """Compute vocabulary overlap between collections."""
    collections = defaultdict(list)
    for doc in docs:
        collections[doc.get("collection", "unknown")].append(doc)

    col_vocabs = {}
    col_freqs = {}
    for col, col_docs in collections.items():
        vocab = set()
        freq = Counter()
        for d in col_docs:
            tokens = tokenize(d["text"])
            vocab.update(tokens)
            freq.update(tokens)
        col_vocabs[col] = vocab
        col_freqs[col] = freq

    col_names = sorted(col_vocabs.keys())
    if len(col_names) < 2:
        return {"error": "Need at least 2 collections"}

    # Pairwise overlap
    overlaps = {}
    for i, col_a in enumerate(col_names):
        for j, col_b in enumerate(col_names):
            if j <= i:
                continue
            shared = col_vocabs[col_a] & col_vocabs[col_b]
            union = col_vocabs[col_a] | col_vocabs[col_b]
            only_a = col_vocabs[col_a] - col_vocabs[col_b]
            only_b = col_vocabs[col_b] - col_vocabs[col_a]

            # Top shared terms by combined freq
            shared_freqs = [(t, col_freqs[col_a][t] + col_freqs[col_b][t]) for t in shared]
            shared_freqs.sort(key=lambda x: x[1], reverse=True)

            # Top unique to each
            unique_a_sorted = sorted([(t, col_freqs[col_a][t]) for t in only_a],
                                     key=lambda x: x[1], reverse=True)[:15]
            unique_b_sorted = sorted([(t, col_freqs[col_b][t]) for t in only_b],
                                     key=lambda x: x[1], reverse=True)[:15]

            overlaps[f"{col_a} <-> {col_b}"] = {
                "jaccard_similarity": round(len(shared) / len(union), 4) if union else 0,
                "shared_terms": len(shared),
                "only_in_first": len(only_a),
                "only_in_second": len(only_b),
                "top_shared_terms": [{"term": t, "combined_freq": c} for t, c in shared_freqs[:15]],
                "top_unique_to_first": [{"term": t, "freq": c} for t, c in unique_a_sorted],
                "top_unique_to_second": [{"term": t, "freq": c} for t, c in unique_b_sorted],
            }

    return {
        "collections": {
            col: {"vocab_size": len(col_vocabs[col]), "total_tokens": sum(col_freqs[col].values())}
            for col in col_names
        },
        "pairwise_overlap": overlaps,
    }


# ═══════════════════════════════════════════════════════
# OCR Quality Comparison
# ═══════════════════════════════════════════════════════

def compare_ocr_quality(docs: list[dict]) -> dict:
    """Compare OCR quality distributions across collections."""
    collections = defaultdict(list)
    for doc in docs:
        collections[doc.get("collection", "unknown")].append(doc)

    import statistics as stats

    result = {}
    for col, col_docs in sorted(collections.items()):
        scores = [d["ocr_quality"] for d in col_docs if d.get("ocr_quality") is not None]
        tiers = Counter(d.get("ocr_quality_tier", "unknown") for d in col_docs)

        result[col] = {
            "n_documents": len(col_docs),
            "mean_ocr": round(stats.mean(scores), 4) if scores else 0,
            "median_ocr": round(stats.median(scores), 4) if scores else 0,
            "std_ocr": round(stats.stdev(scores), 4) if len(scores) > 1 else 0,
            "min_ocr": round(min(scores), 4) if scores else 0,
            "max_ocr": round(max(scores), 4) if scores else 0,
            "tiers": dict(sorted(tiers.items())),
        }

    return result


# ═══════════════════════════════════════════════════════
# Cross-Collection Retrieval
# ═══════════════════════════════════════════════════════

def cross_collection_retrieval(docs: list[dict],
                                model_name: str = "BAAI/bge-small-en-v1.5") -> dict:
    """Test retrieval across collections: queries from one, search in other."""
    from sentence_transformers import SentenceTransformer
    import faiss
    import gc

    model = SentenceTransformer(model_name)

    collections = defaultdict(list)
    for doc in docs:
        collections[doc.get("collection", "unknown")].append(doc)
    col_names = sorted(collections.keys())

    # Generate representative queries from each collection (top TF-IDF terms)
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "it", "that", "this", "was", "are",
        "be", "have", "has", "had", "not", "as", "if", "his", "her", "their",
        "which", "who", "whom", "will", "would", "shall", "should", "may",
        "can", "could", "he", "she", "they", "we", "you", "i", "been", "were",
    }

    n_all = len(docs)
    all_doc_vocabs = [set(tokenize(d["text"])) for d in docs]
    doc_freq = Counter()
    for v in all_doc_vocabs:
        for t in v:
            doc_freq[t] += 1

    col_queries = {}
    for col in col_names:
        col_tf = Counter()
        for d in collections[col]:
            col_tf.update(tokenize(d["text"]))
        total = sum(col_tf.values())
        tfidf = {}
        for term, count in col_tf.items():
            if term in stopwords or len(term) < 4:
                continue
            tf = count / total
            idf = math.log(n_all / (1 + doc_freq.get(term, 0)))
            tfidf[term] = tf * idf

        top = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:10]
        col_queries[col] = [
            " ".join([t[0] for t in top[i:i+3]])
            for i in range(0, min(9, len(top)), 3)
        ]

    # Encode all documents per collection
    col_embeddings = {}
    col_doc_ids = {}
    for col in col_names:
        texts = [d.get("text", "")[:8192] for d in collections[col]]
        print(f"  Encoding {len(texts)} docs for collection '{col}'...")
        embs = model.encode(texts, batch_size=32, show_progress_bar=False,
                           normalize_embeddings=True)
        col_embeddings[col] = embs
        col_doc_ids[col] = [d["doc_id"] for d in collections[col]]

    # Cross-collection retrieval matrix
    results = {}
    k = 10
    for source_col in col_names:
        queries = col_queries[source_col]
        q_embs = model.encode(queries, normalize_embeddings=True)

        results[source_col] = {"queries": queries, "retrieval_from": {}}

        for target_col in col_names:
            # Build FAISS index for target
            dim = col_embeddings[target_col].shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(col_embeddings[target_col].astype(np.float32))

            scores, indices = index.search(q_embs.astype(np.float32), min(k, len(col_doc_ids[target_col])))
            mean_score = float(np.mean(scores[:, 0]))

            results[source_col]["retrieval_from"][target_col] = {
                "mean_top1_similarity": round(mean_score, 4),
                "mean_topk_similarity": round(float(np.mean(scores)), 4),
            }

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

    # ── Figure: Cross-Collection Comparison Dashboard ──
    ocr = results["ocr_comparison"]
    vocab = results["vocabulary_overlap"]

    col_names = sorted(ocr.keys())
    short_names = [n.replace("_and_new_imperialism", "").replace("_", " ").title() for n in col_names]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) OCR quality comparison
    means = [ocr[c]["mean_ocr"] for c in col_names]
    stds = [ocr[c]["std_ocr"] for c in col_names]
    axes[0, 0].bar(short_names, means, yerr=stds, color=["#4C72B0", "#55A868"],
                   alpha=0.8, capsize=5, edgecolor="white")
    axes[0, 0].set_title("(a) Mean OCR Quality (±σ)")
    axes[0, 0].set_ylim(0.85, 1.0)

    # (b) Document count + tier distribution
    for i, col in enumerate(col_names):
        tiers = ocr[col]["tiers"]
        bottom = 0
        colors = {"high": "#55A868", "medium": "#C44E52", "low": "#DD8452"}
        for tier in ["high", "medium", "low"]:
            count = tiers.get(tier, 0)
            axes[0, 1].bar(short_names[i], count, bottom=bottom,
                          color=colors.get(tier, "#8C8C8C"), label=tier if i == 0 else "",
                          alpha=0.8, edgecolor="white")
            bottom += count
    axes[0, 1].set_title("(b) OCR Tier Distribution")
    axes[0, 1].legend()

    # (c) Vocabulary sizes
    vocab_data = vocab.get("collections", {})
    vsizes = [vocab_data.get(c, {}).get("vocab_size", 0) for c in col_names]
    axes[1, 0].bar(short_names, vsizes, color=["#4C72B0", "#55A868"], alpha=0.8, edgecolor="white")
    axes[1, 0].set_title("(c) Vocabulary Size")
    from matplotlib.ticker import FuncFormatter
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))

    # (d) Cross-retrieval similarity
    cross = results.get("cross_collection_retrieval", {})
    if cross:
        matrix_data = []
        labels = []
        for source in col_names:
            row = []
            labels.append(source.replace("_and_new_imperialism", "")[:10])
            for target in col_names:
                val = cross.get(source, {}).get("retrieval_from", {}).get(target, {}).get("mean_top1_similarity", 0)
                row.append(val)
            matrix_data.append(row)

        im = axes[1, 1].imshow(matrix_data, cmap="Blues", vmin=0)
        axes[1, 1].set_xticks(range(len(short_names)))
        axes[1, 1].set_yticks(range(len(short_names)))
        axes[1, 1].set_xticklabels(short_names, rotation=45, ha="right")
        axes[1, 1].set_yticklabels(short_names)
        for i in range(len(short_names)):
            for j in range(len(short_names)):
                axes[1, 1].text(j, i, f"{matrix_data[i][j]:.3f}",
                               ha="center", va="center", fontsize=11)
        axes[1, 1].set_title("(d) Cross-Collection Retrieval (cosine)")
        axes[1, 1].set_xlabel("Target Collection")
        axes[1, 1].set_ylabel("Query Source")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig12_cross_collection.png", bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved fig12_cross_collection.png")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cross-Collection Comparative Analysis")
    parser.add_argument("--corpus", type=str, default="data/corpus.jsonl")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading corpus...")
    docs = load_corpus(corpus_path)
    print(f"  {len(docs)} documents loaded")

    col_counts = Counter(d.get("collection", "unknown") for d in docs)
    print(f"  Collections: {dict(col_counts)}")

    print("\nComparing OCR quality across collections...")
    ocr_comp = compare_ocr_quality(docs)
    for col, data in ocr_comp.items():
        print(f"  {col}: mean={data['mean_ocr']:.4f}, median={data['median_ocr']:.4f}")
        print(f"    Tiers: {data['tiers']}")

    print("\nComputing vocabulary overlap...")
    vocab_overlap = compute_vocabulary_overlap(docs)
    for pair, data in vocab_overlap["pairwise_overlap"].items():
        print(f"  {pair}: Jaccard={data['jaccard_similarity']:.4f}")
        print(f"    Shared: {data['shared_terms']}, Unique-A: {data['only_in_first']}, Unique-B: {data['only_in_second']}")

    print("\nRunning cross-collection retrieval...")
    cross_ret = cross_collection_retrieval(docs)
    for source, data in cross_ret.items():
        for target, scores in data["retrieval_from"].items():
            label = "WITHIN" if source == target else "ACROSS"
            print(f"  {source} → {target} [{label}]: top1={scores['mean_top1_similarity']:.4f}")

    results = {
        "experiment": "Experiment 5: Cross-Collection Comparative Analysis",
        "corpus_size": len(docs),
        "collection_counts": dict(col_counts),
        "ocr_comparison": ocr_comp,
        "vocabulary_overlap": vocab_overlap,
        "cross_collection_retrieval": cross_ret,
    }

    output_file = output_dir / "experiment5_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")

    print("\nGenerating figures...")
    generate_figures(results, output_dir)

    print("\n" + "=" * 60)
    print("Experiment 5 Complete: Cross-Collection Comparative Analysis")
    print("=" * 60)


if __name__ == "__main__":
    main()
