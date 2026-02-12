"""
Experiment 4: Temporal Retrieval Analysis.

Analyzes the impact of temporal span on retrieval effectiveness. Measures
lexical drift across centuries, evaluates BM25 and dense retrieval per
temporal era, and tests cross-temporal retrieval capabilities.

This is a key differentiator for the NLS corpus: few datasets span 5+ centuries.

Usage:
    python experiments/experiment4_temporal.py --corpus data/corpus.jsonl --output results/
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


# ═══════════════════════════════════════════════════════
# Data Loading & Tokenization
# ═══════════════════════════════════════════════════════

def load_corpus(corpus_path: Path) -> list[dict]:
    """Load corpus as list of records."""
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def tokenize(text: str) -> list[str]:
    """Simple tokenizer, lowercased."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def get_era(date_numeric: int | None) -> str:
    """Map document date to historical era."""
    if date_numeric is None:
        return "unknown"
    if date_numeric < 1700:
        return "pre-1700"
    elif date_numeric < 1800:
        return "1700-1799"
    elif date_numeric < 1850:
        return "1800-1849"
    elif date_numeric < 1900:
        return "1850-1899"
    else:
        return "1900+"


ERA_ORDER = ["pre-1700", "1700-1799", "1800-1849", "1850-1899", "1900+"]


# ═══════════════════════════════════════════════════════
# Lexical Drift Analysis
# ═══════════════════════════════════════════════════════

def compute_lexical_drift(docs: list[dict]) -> dict:
    """Compute vocabulary overlap between temporal eras to measure lexical drift."""
    era_vocabs = defaultdict(set)
    era_freqs = defaultdict(Counter)

    for doc in docs:
        era = get_era(doc.get("date_numeric"))
        if era == "unknown":
            continue
        tokens = tokenize(doc["text"])
        era_vocabs[era].update(tokens)
        era_freqs[era].update(tokens)

    # Pairwise Jaccard similarity between eras
    eras = [e for e in ERA_ORDER if e in era_vocabs]
    pairwise = {}
    for i, era_a in enumerate(eras):
        for j, era_b in enumerate(eras):
            if j <= i:
                continue
            intersection = era_vocabs[era_a] & era_vocabs[era_b]
            union = era_vocabs[era_a] | era_vocabs[era_b]
            jaccard = len(intersection) / len(union) if union else 0
            pairwise[f"{era_a} <-> {era_b}"] = {
                "jaccard_similarity": round(jaccard, 4),
                "shared_terms": len(intersection),
                "union_terms": len(union),
                "unique_to_first": len(era_vocabs[era_a] - era_vocabs[era_b]),
                "unique_to_second": len(era_vocabs[era_b] - era_vocabs[era_a]),
            }

    # Era-specific vocabulary stats
    era_stats = {}
    for era in eras:
        era_stats[era] = {
            "vocabulary_size": len(era_vocabs[era]),
            "total_tokens": sum(era_freqs[era].values()),
            "ttr": round(len(era_vocabs[era]) / sum(era_freqs[era].values()), 4) if era_freqs[era] else 0,
        }

    # Drift metric: Jaccard dissimilarity between consecutive eras
    consecutive_drift = []
    for i in range(len(eras) - 1):
        pair_key = f"{eras[i]} <-> {eras[i+1]}"
        if pair_key in pairwise:
            consecutive_drift.append({
                "pair": pair_key,
                "drift": round(1 - pairwise[pair_key]["jaccard_similarity"], 4),
            })

    # Era-unique terms (terms appearing ONLY in that era)
    all_terms = set()
    for v in era_vocabs.values():
        all_terms.update(v)

    era_unique_terms = {}
    for era in eras:
        other_terms = set()
        for other_era, vocab in era_vocabs.items():
            if other_era != era:
                other_terms.update(vocab)
        unique = era_vocabs[era] - other_terms
        # Sort by frequency
        unique_sorted = sorted(
            [(t, era_freqs[era][t]) for t in unique],
            key=lambda x: x[1], reverse=True
        )[:20]
        era_unique_terms[era] = [
            {"term": t, "count": c} for t, c in unique_sorted
        ]

    return {
        "era_vocabulary_stats": era_stats,
        "pairwise_similarity": pairwise,
        "consecutive_drift": consecutive_drift,
        "era_unique_terms": era_unique_terms,
    }


# ═══════════════════════════════════════════════════════
# BM25 Retrieval Per Era
# ═══════════════════════════════════════════════════════

def bm25_score(query_tokens: list[str], doc_tokens: list[str],
               doc_freqs: dict, n_docs: int, avg_dl: float,
               k1: float = 1.2, b: float = 0.75) -> float:
    """Score single document with BM25."""
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


def bm25_retrieval_per_era(docs: list[dict], queries: list[str], k: int = 10) -> dict:
    """Run BM25 retrieval within each era."""
    # Group docs by era
    era_docs = defaultdict(list)
    for doc in docs:
        era = get_era(doc.get("date_numeric"))
        if era != "unknown":
            era_docs[era].append(doc)

    results = {}
    for era in ERA_ORDER:
        if era not in era_docs:
            continue

        edocs = era_docs[era]
        # Precompute
        doc_tokens_list = [tokenize(d["text"]) for d in edocs]
        doc_freqs = Counter()
        for tokens in doc_tokens_list:
            for t in set(tokens):
                doc_freqs[t] += 1
        n_docs = len(edocs)
        avg_dl = sum(len(t) for t in doc_tokens_list) / n_docs if n_docs > 0 else 1

        era_scores = []
        for query in queries:
            q_tokens = tokenize(query)
            scores_for_q = []
            for i, doc_tokens in enumerate(doc_tokens_list):
                s = bm25_score(q_tokens, doc_tokens, doc_freqs, n_docs, avg_dl)
                scores_for_q.append(s)

            # MRR@k: assume query is somewhat relevant to its era
            sorted_idx = sorted(range(len(scores_for_q)), key=lambda x: scores_for_q[x], reverse=True)
            top_score = scores_for_q[sorted_idx[0]] if sorted_idx else 0
            era_scores.append(top_score)

        results[era] = {
            "n_documents": len(edocs),
            "mean_top_bm25_score": round(sum(era_scores) / len(era_scores), 4) if era_scores else 0,
        }

    return results


# ═══════════════════════════════════════════════════════
# Dense Retrieval Per Era
# ═══════════════════════════════════════════════════════

def dense_retrieval_per_era(docs: list[dict], queries: list[str],
                            model_name: str = "BAAI/bge-small-en-v1.5",
                            k: int = 10) -> dict:
    """Run dense retrieval per temporal era using cosine similarity."""
    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer(model_name)

    # Group by era
    era_docs = defaultdict(list)
    for doc in docs:
        era = get_era(doc.get("date_numeric"))
        if era != "unknown":
            era_docs[era].append(doc)

    # Encode all queries once
    print("  Encoding queries...")
    q_embeddings = model.encode(queries, normalize_embeddings=True)

    results = {}
    for era in ERA_ORDER:
        if era not in era_docs:
            continue

        edocs = era_docs[era]
        doc_texts = [d.get("text", "")[:8192] for d in edocs]

        print(f"  Encoding {len(edocs)} docs for era {era}...")
        doc_embeddings = model.encode(doc_texts, batch_size=32,
                                       show_progress_bar=False,
                                       normalize_embeddings=True)

        # Build FAISS index for this era
        dim = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(doc_embeddings.astype(np.float32))

        # Search
        scores, indices = index.search(q_embeddings.astype(np.float32), min(k, len(edocs)))

        # Mean top-k scores across queries
        mean_top_scores = [float(np.mean(scores[i][scores[i] > 0])) if np.any(scores[i] > 0) else 0
                          for i in range(len(queries))]

        results[era] = {
            "n_documents": len(edocs),
            "mean_top_similarity": round(sum(mean_top_scores) / len(mean_top_scores), 4) if mean_top_scores else 0,
            "mean_top1_similarity": round(float(np.mean(scores[:, 0])), 4),
        }

    return results


# ═══════════════════════════════════════════════════════
# Cross-Temporal Retrieval
# ═══════════════════════════════════════════════════════

def cross_temporal_retrieval(docs: list[dict],
                              model_name: str = "BAAI/bge-small-en-v1.5") -> dict:
    """Test if queries from one era can retrieve docs from another era.
    
    For each era, generate 'representative' queries (top TF-IDF terms)
    and search across all eras.
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer(model_name)

    # Group by era
    era_docs = defaultdict(list)
    for doc in docs:
        era = get_era(doc.get("date_numeric"))
        if era != "unknown":
            era_docs[era].append(doc)

    eras = [e for e in ERA_ORDER if e in era_docs]

    # Generate representative queries from each era using top TF-IDF terms
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "it", "that", "this", "was", "are",
        "be", "have", "has", "had", "not", "as", "if", "his", "her", "their",
        "which", "who", "whom", "will", "would", "shall", "should", "may",
        "can", "could", "he", "she", "they", "we", "you", "i", "been", "were",
    }

    # Compute document frequencies across all docs
    all_doc_vocabs = [set(tokenize(d["text"])) for d in docs if get_era(d.get("date_numeric")) != "unknown"]
    doc_freq = Counter()
    for v in all_doc_vocabs:
        for t in v:
            doc_freq[t] += 1
    n_docs_total = len(all_doc_vocabs)

    era_queries = {}
    for era in eras:
        edocs = era_docs[era]
        era_tf = Counter()
        for d in edocs:
            era_tf.update(tokenize(d["text"]))

        total = sum(era_tf.values())
        tfidf = {}
        for term, count in era_tf.items():
            if term in stopwords or len(term) < 4:
                continue
            tf = count / total
            idf = math.log(n_docs_total / (1 + doc_freq.get(term, 0)))
            tfidf[term] = tf * idf

        top_terms = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:5]
        # Create synthetic queries from top terms
        era_queries[era] = [" ".join([t[0] for t in top_terms[:3]]),
                            " ".join([t[0] for t in top_terms[1:4]]),
                            " ".join([t[0] for t in top_terms[2:5]])]

    # Encode all docs
    all_doc_texts = []
    all_doc_eras = []
    for era in eras:
        for d in era_docs[era]:
            all_doc_texts.append(d.get("text", "")[:8192])
            all_doc_eras.append(era)

    print("  Encoding all documents for cross-temporal analysis...")
    doc_embeddings = model.encode(all_doc_texts, batch_size=32,
                                  show_progress_bar=True,
                                  normalize_embeddings=True)

    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings.astype(np.float32))

    # For each source era, search and see which era the top results come from
    retrieval_matrix = {source: {target: 0 for target in eras} for source in eras}
    k = 10

    for source_era, queries in era_queries.items():
        q_emb = model.encode(queries, normalize_embeddings=True)
        scores, indices = index.search(q_emb.astype(np.float32), k)

        for i in range(len(queries)):
            for j in range(k):
                if indices[i][j] >= 0:
                    retrieved_era = all_doc_eras[indices[i][j]]
                    retrieval_matrix[source_era][retrieved_era] += 1

    # Normalize to proportions
    for source in eras:
        total = sum(retrieval_matrix[source].values())
        if total > 0:
            for target in eras:
                retrieval_matrix[source][target] = round(
                    retrieval_matrix[source][target] / total, 4
                )

    return {
        "era_queries": {era: qs for era, qs in era_queries.items()},
        "retrieval_matrix": retrieval_matrix,
    }


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

    # ── Figure: Lexical Drift Heatmap ──
    drift = results["lexical_drift"]
    pairwise = drift["pairwise_similarity"]
    eras = list(drift["era_vocabulary_stats"].keys())

    heat = np.zeros((len(eras), len(eras)))
    for i, ea in enumerate(eras):
        heat[i][i] = 1.0  # Self-similarity
        for j, eb in enumerate(eras):
            if j > i:
                key = f"{ea} <-> {eb}"
                if key in pairwise:
                    val = pairwise[key]["jaccard_similarity"]
                    heat[i][j] = val
                    heat[j][i] = val

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heat, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(eras)))
    ax.set_yticks(range(len(eras)))
    ax.set_xticklabels(eras, rotation=45, ha="right")
    ax.set_yticklabels(eras)
    for i in range(len(eras)):
        for j in range(len(eras)):
            ax.text(j, i, f"{heat[i][j]:.2f}", ha="center", va="center",
                    color="white" if heat[i][j] > 0.5 else "black", fontsize=10)
    plt.colorbar(im, label="Jaccard Similarity")
    ax.set_title("Vocabulary Overlap (Jaccard Similarity) Across Temporal Eras")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig9_lexical_drift_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved fig9_lexical_drift_heatmap.png")

    # ── Figure: Cross-Temporal Retrieval Matrix ──
    cross = results.get("cross_temporal_retrieval", {})
    if cross and "retrieval_matrix" in cross:
        matrix = cross["retrieval_matrix"]
        eras_m = [e for e in ERA_ORDER if e in matrix]

        heat2 = np.zeros((len(eras_m), len(eras_m)))
        for i, source in enumerate(eras_m):
            for j, target in enumerate(eras_m):
                heat2[i][j] = matrix[source].get(target, 0)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heat2, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(eras_m)))
        ax.set_yticks(range(len(eras_m)))
        ax.set_xticklabels(eras_m, rotation=45, ha="right")
        ax.set_yticklabels(eras_m)
        for i in range(len(eras_m)):
            for j in range(len(eras_m)):
                ax.text(j, i, f"{heat2[i][j]:.2f}", ha="center", va="center",
                        color="white" if heat2[i][j] > 0.4 else "black", fontsize=10)
        plt.colorbar(im, label="Proportion of Retrieved Documents")
        ax.set_xlabel("Retrieved Era")
        ax.set_ylabel("Query Source Era")
        ax.set_title("Cross-Temporal Dense Retrieval Matrix")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig10_cross_temporal_retrieval.png", bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved fig10_cross_temporal_retrieval.png")

    # ── Figure: Dense Retrieval Scores Per Era ──
    dense = results.get("dense_retrieval_per_era", {})
    if dense:
        fig, ax = plt.subplots(figsize=(8, 5))
        eras_d = [e for e in ERA_ORDER if e in dense]
        similarities = [dense[e]["mean_top1_similarity"] for e in eras_d]
        ax.bar(eras_d, similarities, color="#4C72B0", alpha=0.8, edgecolor="white")
        ax.set_xlabel("Temporal Era")
        ax.set_ylabel("Mean Top-1 Similarity")
        ax.set_title("Dense Retrieval Effectiveness by Era")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig11_retrieval_by_era.png", bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved fig11_retrieval_by_era.png")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Temporal Retrieval Analysis")
    parser.add_argument("--corpus", type=str, default="data/corpus.jsonl")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading corpus...")
    docs = load_corpus(corpus_path)
    print(f"  {len(docs)} documents loaded")

    # Group by era
    era_counts = Counter(get_era(d.get("date_numeric")) for d in docs)
    print("\n  Documents by era:")
    for era in ERA_ORDER:
        if era in era_counts:
            print(f"    {era}: {era_counts[era]}")

    print("\nComputing lexical drift across eras...")
    lexical_drift = compute_lexical_drift(docs)
    for pair, data in lexical_drift["pairwise_similarity"].items():
        print(f"  {pair}: Jaccard = {data['jaccard_similarity']:.4f}")
    for d in lexical_drift["consecutive_drift"]:
        print(f"  Drift: {d['pair']}: {d['drift']:.4f}")

    # Generate some test queries from the corpus for per-era retrieval
    print("\nGenerating temporal test queries...")
    test_queries = [
        "colonial administration India",
        "trade commerce East India Company",
        "missionary exploration Africa",
        "correspondence official letters",
        "political negotiations treaty",
        "military expedition campaign",
        "natural history specimens botanical",
        "railway construction engineering",
        "tribal customs traditional practices",
        "legislative council governance",
    ]

    print("\nRunning BM25 retrieval per era...")
    bm25_per_era = bm25_retrieval_per_era(docs, test_queries)
    for era, data in bm25_per_era.items():
        print(f"  {era}: mean top BM25 = {data['mean_top_bm25_score']:.4f} ({data['n_documents']} docs)")

    print("\nRunning dense retrieval per era...")
    dense_per_era = dense_retrieval_per_era(docs, test_queries)
    for era, data in dense_per_era.items():
        print(f"  {era}: mean top-1 sim = {data['mean_top1_similarity']:.4f} ({data['n_documents']} docs)")

    print("\nRunning cross-temporal retrieval analysis...")
    cross_temporal = cross_temporal_retrieval(docs)
    print("  Retrieval matrix (source era → retrieved era):")
    for source, targets in cross_temporal["retrieval_matrix"].items():
        print(f"    {source}: {targets}")

    # Combine results
    results = {
        "experiment": "Experiment 4: Temporal Retrieval Analysis",
        "corpus_size": len(docs),
        "era_distribution": dict(era_counts),
        "lexical_drift": lexical_drift,
        "bm25_per_era": bm25_per_era,
        "dense_retrieval_per_era": dense_per_era,
        "cross_temporal_retrieval": cross_temporal,
        "test_queries": test_queries,
    }

    output_file = output_dir / "experiment4_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")

    print("\nGenerating figures...")
    generate_figures(results, output_dir)

    print("\n" + "=" * 60)
    print("Experiment 4 Complete: Temporal Retrieval Analysis")
    print("=" * 60)


if __name__ == "__main__":
    main()
