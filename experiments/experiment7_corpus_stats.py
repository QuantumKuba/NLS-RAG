"""
Experiment 7: Corpus Statistics & Distributional Analysis.

Computes vocabulary richness metrics, word frequency distributions (Zipf's law),
OCR quality distributions, document length analysis, and TF-IDF term analysis
per collection and temporal era. Essential descriptive statistics for the
SIGIR resource paper.

Usage:
    python experiments/experiment7_corpus_stats.py --corpus data/corpus.jsonl --output results/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════

def load_corpus(corpus_path: Path) -> list[dict]:
    """Load corpus as list of records."""
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


# ═══════════════════════════════════════════════════════
# Vocabulary & Distributional Analysis
# ═══════════════════════════════════════════════════════

def compute_vocabulary_stats(docs: list[dict]) -> dict:
    """Compute vocabulary richness metrics."""
    all_tokens = []
    doc_vocabs = []

    for doc in docs:
        tokens = tokenize(doc["text"])
        all_tokens.extend(tokens)
        doc_vocabs.append(set(tokens))

    freq = Counter(all_tokens)
    total_tokens = len(all_tokens)
    vocab_size = len(freq)

    # Type-Token Ratio
    ttr = vocab_size / total_tokens if total_tokens > 0 else 0

    # Hapax legomena (words occurring exactly once)
    hapax = sum(1 for w, c in freq.items() if c == 1)
    hapax_ratio = hapax / vocab_size if vocab_size > 0 else 0

    # Dis legomena (words occurring exactly twice)
    dis = sum(1 for w, c in freq.items() if c == 2)

    # Per-document TTR
    doc_ttrs = []
    for doc in docs:
        tokens = tokenize(doc["text"])
        if len(tokens) > 0:
            doc_ttrs.append(len(set(tokens)) / len(tokens))

    return {
        "total_tokens": total_tokens,
        "vocabulary_size": vocab_size,
        "type_token_ratio": round(ttr, 6),
        "hapax_legomena": hapax,
        "hapax_ratio": round(hapax_ratio, 4),
        "dis_legomena": dis,
        "top_50_terms": [
            {"term": w, "count": c, "frequency": round(c / total_tokens, 6)}
            for w, c in freq.most_common(50)
        ],
        "doc_ttr_mean": round(statistics.mean(doc_ttrs), 4) if doc_ttrs else 0,
        "doc_ttr_std": round(statistics.stdev(doc_ttrs), 4) if len(doc_ttrs) > 1 else 0,
        "doc_ttr_median": round(statistics.median(doc_ttrs), 4) if doc_ttrs else 0,
    }


def compute_zipf_analysis(docs: list[dict]) -> dict:
    """Analyze compliance with Zipf's law (frequency ∝ 1/rank)."""
    all_tokens = []
    for doc in docs:
        all_tokens.extend(tokenize(doc["text"]))

    freq = Counter(all_tokens)
    sorted_freqs = sorted(freq.values(), reverse=True)

    # Compute log-log relationship
    ranks = list(range(1, len(sorted_freqs) + 1))

    # Fit linear regression on log-log scale (first 1000 terms)
    n = min(1000, len(sorted_freqs))
    log_ranks = [math.log(r) for r in ranks[:n]]
    log_freqs = [math.log(f) for f in sorted_freqs[:n]]

    # Simple linear regression
    mean_x = sum(log_ranks) / n
    mean_y = sum(log_freqs) / n
    ss_xy = sum((log_ranks[i] - mean_x) * (log_freqs[i] - mean_y) for i in range(n))
    ss_xx = sum((log_ranks[i] - mean_x) ** 2 for i in range(n))
    slope = ss_xy / ss_xx if ss_xx > 0 else 0
    intercept = mean_y - slope * mean_x

    # R² (coefficient of determination)
    ss_res = sum((log_freqs[i] - (slope * log_ranks[i] + intercept)) ** 2 for i in range(n))
    ss_tot = sum((log_freqs[i] - mean_y) ** 2 for i in range(n))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Sample points for plotting
    sample_points = []
    for i in [0, 1, 2, 4, 9, 24, 49, 99, 249, 499, 999, 2499, 4999, 9999]:
        if i < len(sorted_freqs):
            sample_points.append({"rank": i + 1, "frequency": sorted_freqs[i]})

    return {
        "zipf_slope": round(slope, 4),
        "zipf_intercept": round(intercept, 4),
        "zipf_r_squared": round(r_squared, 4),
        "ideal_slope": -1.0,
        "sample_points": sample_points,
        "n_terms_fitted": n,
    }


# ═══════════════════════════════════════════════════════
# OCR Quality Distribution
# ═══════════════════════════════════════════════════════

def compute_ocr_distribution(docs: list[dict]) -> dict:
    """Compute OCR quality distribution statistics."""
    scores = [d["ocr_quality"] for d in docs if d.get("ocr_quality") is not None]
    tiers = Counter(d.get("ocr_quality_tier", "unknown") for d in docs)

    if not scores:
        return {"n_documents": len(docs), "scores": [], "tiers": dict(tiers)}

    # Histogram bins
    bins = np.arange(0.5, 1.01, 0.025)
    hist, bin_edges = np.histogram(scores, bins=bins)
    histogram = [
        {
            "bin_start": round(float(bin_edges[i]), 3),
            "bin_end": round(float(bin_edges[i + 1]), 3),
            "count": int(hist[i]),
        }
        for i in range(len(hist))
    ]

    return {
        "n_documents": len(docs),
        "n_with_scores": len(scores),
        "mean": round(statistics.mean(scores), 4),
        "median": round(statistics.median(scores), 4),
        "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0,
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "percentiles": {
            "P10": round(float(np.percentile(scores, 10)), 4),
            "P25": round(float(np.percentile(scores, 25)), 4),
            "P50": round(float(np.percentile(scores, 50)), 4),
            "P75": round(float(np.percentile(scores, 75)), 4),
            "P90": round(float(np.percentile(scores, 90)), 4),
        },
        "tiers": {k: v for k, v in sorted(tiers.items())},
        "histogram": histogram,
    }


# ═══════════════════════════════════════════════════════
# Document Length Analysis
# ═══════════════════════════════════════════════════════

def compute_length_analysis(docs: list[dict]) -> dict:
    """Compute document length distribution and length-OCR correlation."""
    lengths = [d.get("word_count", len(d.get("text", "").split())) for d in docs]
    scores = [d.get("ocr_quality", 0) for d in docs]

    # Length distribution
    length_stats = {
        "n_documents": len(docs),
        "mean": round(statistics.mean(lengths), 1),
        "median": round(statistics.median(lengths), 1),
        "std": round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0,
        "min": min(lengths),
        "max": max(lengths),
        "percentiles": {
            f"P{p}": int(np.percentile(lengths, p))
            for p in [10, 25, 50, 75, 90, 95, 99]
        },
    }

    # Length bins for analysis
    bins_def = [
        ("0-1K", 0, 1000),
        ("1K-5K", 1000, 5000),
        ("5K-20K", 5000, 20000),
        ("20K-50K", 20000, 50000),
        ("50K-100K", 50000, 100000),
        ("100K+", 100000, float("inf")),
    ]
    length_bins = []
    for label, lo, hi in bins_def:
        bin_docs = [d for d, l in zip(docs, lengths) if lo <= l < hi]
        bin_scores = [d.get("ocr_quality", 0) for d in bin_docs]
        length_bins.append({
            "bin": label,
            "count": len(bin_docs),
            "mean_ocr_quality": round(statistics.mean(bin_scores), 4) if bin_scores else 0,
        })

    # Pearson correlation: length vs OCR quality
    if len(lengths) > 2:
        mean_l = statistics.mean(lengths)
        mean_s = statistics.mean(scores)
        cov = sum((lengths[i] - mean_l) * (scores[i] - mean_s) for i in range(len(lengths))) / len(lengths)
        std_l = statistics.stdev(lengths)
        std_s = statistics.stdev(scores)
        correlation = cov / (std_l * std_s) if std_l > 0 and std_s > 0 else 0
    else:
        correlation = 0

    return {
        "length_distribution": length_stats,
        "length_bins": length_bins,
        "length_ocr_correlation": round(correlation, 4),
    }


# ═══════════════════════════════════════════════════════
# TF-IDF Analysis Per Collection and Era
# ═══════════════════════════════════════════════════════

def compute_tfidf_terms(docs: list[dict], group_key: str, top_n: int = 15) -> dict:
    """Compute top TF-IDF terms per group (collection or era)."""
    # Group documents
    groups = defaultdict(list)
    for doc in docs:
        key = doc.get(group_key, "unknown")
        if group_key == "era":
            key = _get_era(doc)
        groups[key].append(doc)

    # Stopwords (minimal set)
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "it", "that", "this", "was", "are",
        "be", "have", "has", "had", "not", "as", "if", "his", "her", "their",
        "which", "who", "whom", "will", "would", "shall", "should", "may",
        "can", "could", "he", "she", "they", "we", "you", "i", "my", "your",
        "our", "its", "been", "were", "do", "did", "he", "him", "them", "all",
        "no", "so", "than", "when", "what", "there", "also", "about", "after",
        "upon", "more", "other", "one", "two", "three", "said", "such", "same",
        "very", "most", "much", "some", "any", "up", "out", "into", "over",
    }

    # Document frequency across all groups (for IDF)
    all_doc_vocabs = []
    for doc in docs:
        tokens = set(tokenize(doc["text"]))
        all_doc_vocabs.append(tokens)

    n_docs = len(docs)
    doc_freq = Counter()
    for vocab in all_doc_vocabs:
        for w in vocab:
            doc_freq[w] += 1

    results = {}
    for group_name, group_docs in sorted(groups.items()):
        # Term frequency within group
        group_tf = Counter()
        for doc in group_docs:
            group_tf.update(tokenize(doc["text"]))

        # TF-IDF scoring
        tfidf_scores = {}
        total_terms = sum(group_tf.values())
        for term, count in group_tf.items():
            if term in stopwords or len(term) < 3:
                continue
            tf = count / total_terms
            idf = math.log(n_docs / (1 + doc_freq.get(term, 0)))
            tfidf_scores[term] = tf * idf

        top_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        results[group_name] = {
            "n_documents": len(group_docs),
            "top_terms": [
                {"term": t, "tfidf": round(s, 6), "count": group_tf[t]}
                for t, s in top_terms
            ],
        }

    return results


def _get_era(doc: dict) -> str:
    """Map document date to historical era."""
    date = doc.get("date_numeric")
    if date is None:
        return "unknown"
    if date < 1700:
        return "pre-1700"
    elif date < 1800:
        return "1700-1799"
    elif date < 1850:
        return "1800-1849"
    elif date < 1900:
        return "1850-1899"
    else:
        return "1900+"


# ═══════════════════════════════════════════════════════
# Temporal Coverage
# ═══════════════════════════════════════════════════════

def compute_temporal_coverage(docs: list[dict]) -> dict:
    """Detailed temporal distribution analysis."""
    dates = [d["date_numeric"] for d in docs if d.get("date_numeric")]
    if not dates:
        return {}

    # By decade
    decades = Counter()
    for d in dates:
        decades[f"{(d // 10) * 10}s"] += 1

    # By era
    eras = Counter()
    for d in dates:
        era = _get_era({"date_numeric": d})
        eras[era] += 1

    # By collection × era
    collection_era = defaultdict(lambda: Counter())
    for doc in docs:
        if doc.get("date_numeric"):
            era = _get_era(doc)
            collection_era[doc["collection"]][era] += 1

    return {
        "date_range": {"earliest": min(dates), "latest": max(dates), "span_years": max(dates) - min(dates)},
        "by_decade": dict(sorted(decades.items())),
        "by_era": dict(sorted(eras.items())),
        "by_collection_era": {
            col: dict(sorted(era_counts.items()))
            for col, era_counts in sorted(collection_era.items())
        },
    }


# ═══════════════════════════════════════════════════════
# Per-Collection Breakdown
# ═══════════════════════════════════════════════════════

def compute_collection_stats(docs: list[dict]) -> dict:
    """Compute per-collection statistics."""
    collections = defaultdict(list)
    for doc in docs:
        collections[doc.get("collection", "unknown")].append(doc)

    result = {}
    for col_name, col_docs in sorted(collections.items()):
        lengths = [d.get("word_count", 0) for d in col_docs]
        scores = [d["ocr_quality"] for d in col_docs if d.get("ocr_quality") is not None]
        tiers = Counter(d.get("ocr_quality_tier", "unknown") for d in col_docs)
        dates = [d["date_numeric"] for d in col_docs if d.get("date_numeric")]

        # Vocabulary
        all_tokens = []
        for d in col_docs:
            all_tokens.extend(tokenize(d["text"]))
        vocab_size = len(set(all_tokens))

        result[col_name] = {
            "n_documents": len(col_docs),
            "total_tokens": len(all_tokens),
            "vocabulary_size": vocab_size,
            "type_token_ratio": round(vocab_size / len(all_tokens), 4) if all_tokens else 0,
            "mean_doc_length": round(statistics.mean(lengths), 1) if lengths else 0,
            "median_doc_length": round(statistics.median(lengths), 1) if lengths else 0,
            "mean_ocr_quality": round(statistics.mean(scores), 4) if scores else 0,
            "ocr_tiers": dict(sorted(tiers.items())),
            "date_range": f"{min(dates)}-{max(dates)}" if dates else "N/A",
        }

    return result


# ═══════════════════════════════════════════════════════
# Figure Generation
# ═══════════════════════════════════════════════════════

def generate_figures(results: dict, output_dir: Path):
    """Generate publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
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

    # ── Figure 4: OCR Quality Distribution ──
    ocr = results["ocr_distribution"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Histogram
    bins_data = ocr["histogram"]
    bin_centers = [(b["bin_start"] + b["bin_end"]) / 2 for b in bins_data]
    counts = [b["count"] for b in bins_data]
    widths = [b["bin_end"] - b["bin_start"] for b in bins_data]
    axes[0].bar(bin_centers, counts, width=widths, alpha=0.8, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("Estimated OCR Quality Score")
    axes[0].set_ylabel("Number of Documents")
    axes[0].set_title("(a) OCR Quality Score Distribution")
    axes[0].axvline(x=0.98, color="green", linestyle="--", alpha=0.7, label="High threshold")
    axes[0].axvline(x=0.95, color="orange", linestyle="--", alpha=0.7, label="Medium threshold")
    axes[0].legend(fontsize=9)

    # Tier pie chart
    tiers = ocr["tiers"]
    tier_labels = list(tiers.keys())
    tier_counts = list(tiers.values())
    colors = {"high": "#55A868", "medium": "#C44E52", "low": "#DD8452", "unknown": "#8C8C8C"}
    tier_colors = [colors.get(t, "#8C8C8C") for t in tier_labels]
    axes[1].pie(tier_counts, labels=[f"{l}\n(n={c})" for l, c in zip(tier_labels, tier_counts)],
                autopct="%1.1f%%", colors=tier_colors, startangle=90)
    axes[1].set_title("(b) OCR Quality Tiers")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_ocr_distribution.png", bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved fig4_ocr_distribution.png")

    # ── Figure 5: Zipf's Law ──
    zipf = results["zipf_analysis"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sample = zipf["sample_points"]
    ranks = [p["rank"] for p in sample]
    freqs = [p["frequency"] for p in sample]
    ax.scatter(ranks, freqs, c="#4C72B0", s=40, zorder=3, label="Observed")

    # Fitted line
    x_fit = np.logspace(0, math.log10(max(ranks)), 100)
    y_fit = np.exp(zipf["zipf_intercept"]) * x_fit ** zipf["zipf_slope"]
    ax.plot(x_fit, y_fit, "r--", alpha=0.7, label=f"Fit (slope={zipf['zipf_slope']:.2f}, R²={zipf['zipf_r_squared']:.3f})")

    # Ideal Zipf
    y_ideal = freqs[0] / x_fit
    ax.plot(x_fit, y_ideal, "k:", alpha=0.4, label="Ideal Zipf (slope=-1.0)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    ax.set_title("Word Frequency Distribution (Zipf's Law)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "fig5_zipf_law.png", bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved fig5_zipf_law.png")

    # ── Figure 6: Length vs OCR Quality ──
    length = results["length_analysis"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Length bins bar chart
    bins_data = length["length_bins"]
    bin_labels = [b["bin"] for b in bins_data]
    bin_counts = [b["count"] for b in bins_data]
    bin_ocr = [b["mean_ocr_quality"] for b in bins_data]

    axes[0].bar(bin_labels, bin_counts, color="#4C72B0", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Document Length (words)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("(a) Document Length Distribution")
    axes[0].tick_params(axis="x", rotation=30)

    # OCR by length bin
    axes[1].bar(bin_labels, bin_ocr, color="#55A868", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("Document Length (words)")
    axes[1].set_ylabel("Mean OCR Quality Score")
    axes[1].set_title(f"(b) OCR Quality by Length (r={length['length_ocr_correlation']:.3f})")
    axes[1].set_ylim(0.9, 1.0)
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(fig_dir / "fig6_length_vs_ocr.png", bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved fig6_length_vs_ocr.png")

    # ── Figure 7: Temporal Distribution ──
    temporal = results["temporal_coverage"]
    fig, ax = plt.subplots(figsize=(10, 5))

    decades = temporal["by_decade"]
    x_labels = list(decades.keys())
    x_counts = list(decades.values())
    ax.bar(range(len(x_labels)), x_counts, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Number of Documents")
    ax.set_title("Temporal Distribution of Documents by Decade")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "fig7_temporal_distribution.png", bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved fig7_temporal_distribution.png")

    # ── Figure 8: Collection Comparison ──
    col_stats = results["collection_stats"]
    if len(col_stats) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        col_names = list(col_stats.keys())
        # Shorten names for display
        short_names = [n.replace("_and_new_imperialism", "").replace("_", " ").title() for n in col_names]

        # Doc count
        doc_counts = [col_stats[c]["n_documents"] for c in col_names]
        axes[0].bar(short_names, doc_counts, color=["#4C72B0", "#55A868"], alpha=0.8)
        axes[0].set_title("(a) Documents per Collection")
        axes[0].set_ylabel("Count")

        # Mean OCR
        ocr_means = [col_stats[c]["mean_ocr_quality"] for c in col_names]
        axes[1].bar(short_names, ocr_means, color=["#4C72B0", "#55A868"], alpha=0.8)
        axes[1].set_title("(b) Mean OCR Quality")
        axes[1].set_ylim(0.9, 1.0)

        # Vocabulary size
        vocab_sizes = [col_stats[c]["vocabulary_size"] for c in col_names]
        axes[2].bar(short_names, vocab_sizes, color=["#4C72B0", "#55A868"], alpha=0.8)
        axes[2].set_title("(c) Vocabulary Size")
        axes[2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))

        plt.tight_layout()
        plt.savefig(fig_dir / "fig8_collection_comparison.png", bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved fig8_collection_comparison.png")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Corpus Statistics & Distributional Analysis")
    parser.add_argument("--corpus", type=str, default="data/corpus.jsonl")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading corpus...")
    docs = load_corpus(corpus_path)
    print(f"  {len(docs)} documents loaded")

    print("\nComputing vocabulary statistics...")
    vocab_stats = compute_vocabulary_stats(docs)
    print(f"  Total tokens: {vocab_stats['total_tokens']:,}")
    print(f"  Vocabulary size: {vocab_stats['vocabulary_size']:,}")
    print(f"  Type-Token Ratio: {vocab_stats['type_token_ratio']:.4f}")
    print(f"  Hapax legomena: {vocab_stats['hapax_legomena']:,} ({vocab_stats['hapax_ratio']:.1%})")

    print("\nAnalyzing Zipf's law compliance...")
    zipf_stats = compute_zipf_analysis(docs)
    print(f"  Zipf slope: {zipf_stats['zipf_slope']:.4f} (ideal: -1.0)")
    print(f"  R²: {zipf_stats['zipf_r_squared']:.4f}")

    print("\nComputing OCR quality distribution...")
    ocr_dist = compute_ocr_distribution(docs)
    print(f"  Mean: {ocr_dist['mean']:.4f}, Median: {ocr_dist['median']:.4f}")
    for tier, count in ocr_dist["tiers"].items():
        print(f"    {tier}: {count} ({count/len(docs)*100:.1f}%)")

    print("\nAnalyzing document lengths...")
    length_stats = compute_length_analysis(docs)
    print(f"  Mean: {length_stats['length_distribution']['mean']:,.0f} words")
    print(f"  Length-OCR correlation: r = {length_stats['length_ocr_correlation']:.4f}")

    print("\nComputing TF-IDF terms by collection...")
    tfidf_by_collection = compute_tfidf_terms(docs, "collection")
    for col, data in tfidf_by_collection.items():
        top3 = [t["term"] for t in data["top_terms"][:3]]
        print(f"  {col}: {', '.join(top3)}")

    print("\nComputing TF-IDF terms by era...")
    tfidf_by_era = compute_tfidf_terms(docs, "era")
    for era, data in sorted(tfidf_by_era.items()):
        top3 = [t["term"] for t in data["top_terms"][:3]]
        print(f"  {era} ({data['n_documents']} docs): {', '.join(top3)}")

    print("\nComputing temporal coverage...")
    temporal = compute_temporal_coverage(docs)
    if temporal:
        print(f"  Range: {temporal['date_range']['earliest']}-{temporal['date_range']['latest']} "
              f"({temporal['date_range']['span_years']} years)")

    print("\nComputing per-collection statistics...")
    collection_stats = compute_collection_stats(docs)
    for col, stats in collection_stats.items():
        print(f"  {col}: {stats['n_documents']} docs, "
              f"vocab={stats['vocabulary_size']:,}, "
              f"OCR={stats['mean_ocr_quality']:.4f}")

    # Combine results
    results = {
        "experiment": "Experiment 7: Corpus Statistics & Distributional Analysis",
        "corpus_size": len(docs),
        "vocabulary_stats": vocab_stats,
        "zipf_analysis": zipf_stats,
        "ocr_distribution": ocr_dist,
        "length_analysis": length_stats,
        "tfidf_by_collection": tfidf_by_collection,
        "tfidf_by_era": tfidf_by_era,
        "temporal_coverage": temporal,
        "collection_stats": collection_stats,
    }

    # Save results
    output_file = output_dir / "experiment7_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(results, output_dir)

    print("\n" + "=" * 60)
    print("Experiment 7 Complete: Corpus Statistics & Distributional Analysis")
    print("=" * 60)


if __name__ == "__main__":
    main()
