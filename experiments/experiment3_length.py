"""
Experiment 3: Document Length Challenge.

Analyzes how document length affects retrieval and chunking effectiveness,
showing that standard approaches fail on extreme-length archival documents.

Key finding: Standard chunking loses context coherence on 99th percentile
documents, motivating need for hierarchical retrieval.

Usage:
    python experiments/experiment3_length.py --corpus data/corpus.jsonl
    python experiments/experiment3_length.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np
import statistics
from collections import defaultdict
from pathlib import Path

# Prevent tokenizer parallelism segfaults on ARM Mac / Python 3.9
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_corpus(corpus_path: Path) -> list[dict]:
    """Load corpus as list of records."""
    records = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


# ═══════════════════════════════════════════════════════
# Length Distribution Analysis
# ═══════════════════════════════════════════════════════

def compute_length_percentiles(records: list[dict]) -> dict:
    """Compute document length distribution and identify percentile documents."""
    word_counts = [(r["doc_id"], r.get("word_count", 0)) for r in records]
    word_counts.sort(key=lambda x: x[1])
    
    n = len(word_counts)
    percentiles = {}
    
    for p in [10, 25, 50, 75, 90, 95, 99]:
        idx = min(int(n * p / 100), n - 1)
        doc_id, wc = word_counts[idx]
        percentiles[f"P{p}"] = {
            "doc_id": doc_id,
            "word_count": wc,
            "percentile": p,
        }
    
    # Summary statistics
    all_wc = [wc for _, wc in word_counts]
    distribution = {
        "percentiles": percentiles,
        "stats": {
            "n_documents": n,
            "mean": statistics.mean(all_wc),
            "median": statistics.median(all_wc),
            "std": statistics.stdev(all_wc) if len(all_wc) > 1 else 0,
            "min": min(all_wc),
            "max": max(all_wc),
        },
    }
    
    return distribution


def sample_by_percentile(records: list[dict], 
                          target_percentiles: list[int] = [25, 50, 90, 99],
                          n_per_percentile: int = 5) -> dict:
    """Sample documents around each target percentile."""
    sorted_records = sorted(records, key=lambda r: r.get("word_count", 0))
    n = len(sorted_records)
    
    samples = {}
    for p in target_percentiles:
        center_idx = min(int(n * p / 100), n - 1)
        
        # Take n documents around the center
        start = max(0, center_idx - n_per_percentile // 2)
        end = min(n, start + n_per_percentile)
        
        samples[f"P{p}"] = sorted_records[start:end]
    
    return samples


# ═══════════════════════════════════════════════════════
# Chunking Analysis
# ═══════════════════════════════════════════════════════

def analyze_chunking(text: str, chunk_size: int = 512, overlap: int = 50) -> dict:
    """Analyze how a document gets chunked and what information is lost."""
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        return {"n_chunks": 0, "total_words": 0}
    
    # Generate chunks
    chunks = []
    start = 0
    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "text": chunk_text,
            "start": start,
            "end": end,
            "word_count": end - start,
        })
        start += chunk_size - overlap
    
    # Analyze chunk coherence: measure how many sentences are split across chunks
    sentences = text.split(".")
    split_sentences = 0
    for i in range(len(chunks) - 1):
        boundary_word = chunks[i]["end"]
        # Check if a sentence crosses this boundary
        char_pos = len(" ".join(words[:boundary_word]))
        for sent in sentences:
            sent_start = text.find(sent)
            sent_end = sent_start + len(sent)
            if sent_start < char_pos < sent_end:
                split_sentences += 1
                break
    
    # Calculate context coverage: how much of the document is retrievable
    # with top-k chunks (simulate retrieval keeping only k chunks)
    for k in [1, 3, 5, 10]:
        coverage = min(k * chunk_size, total_words) / total_words
    
    return {
        "total_words": total_words,
        "n_chunks": len(chunks),
        "avg_chunk_length": total_words / len(chunks) if chunks else 0,
        "split_sentences_estimate": split_sentences,
        "coverage_at_k": {
            k: min(k * chunk_size / total_words, 1.0) 
            for k in [1, 3, 5, 10]
        },
    }


def compare_retrieval_strategies(records: list[dict], percentile_samples: dict,
                                  output_dir: Path) -> dict:
    """Compare full-doc vs passage-level retrieval at each percentile."""
    from sentence_transformers import SentenceTransformer
    
    print("Loading embedding model...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    results = {}
    
    for p_label, docs in percentile_samples.items():
        print(f"\n  Analyzing {p_label} ({len(docs)} documents)...")
        
        percentile_results = []
        
        for doc in docs:
            text = doc.get("text", "")
            word_count = doc.get("word_count", 0)
            
            if not text.strip():
                continue
            
            # ─── Strategy 1: Full document embedding ───
            # Truncate to model max length
            truncated_text = " ".join(text.split()[:8192])
            full_doc_emb = model.encode([truncated_text], normalize_embeddings=True)
            
            # How much of the document is captured?
            full_doc_coverage = min(8192, word_count) / word_count if word_count > 0 else 0
            
            # ─── Strategy 2: Passage-level chunking (512 tokens) ───
            chunk_analysis = analyze_chunking(text, chunk_size=512)
            
            # Encode a sample of chunks (up to 20 for efficiency)
            chunks = []
            words = text.split()
            start = 0
            while start < len(words) and len(chunks) < 50:
                end = min(start + 512, len(words))
                chunks.append(" ".join(words[start:end]))
                start += 462  # 512 - 50 overlap
            
            if chunks:
                chunk_embs = model.encode(chunks, normalize_embeddings=True)
                
                # Self-similarity between chunks (how coherent are they?)
                if len(chunk_embs) > 1:
                    sim_matrix = np.dot(chunk_embs, chunk_embs.T)
                    # Mean off-diagonal similarity
                    mask = ~np.eye(len(chunk_embs), dtype=bool)
                    mean_inter_chunk_sim = float(np.mean(sim_matrix[mask]))
                else:
                    mean_inter_chunk_sim = 1.0
                
                # How similar is each chunk to the full document?
                chunk_to_doc_sims = np.dot(chunk_embs, full_doc_emb.T).squeeze()
                mean_chunk_doc_sim = float(np.mean(chunk_to_doc_sims))
                max_chunk_doc_sim = float(np.max(chunk_to_doc_sims))
            else:
                mean_inter_chunk_sim = 0
                mean_chunk_doc_sim = 0
                max_chunk_doc_sim = 0
            
            doc_result = {
                "doc_id": doc["doc_id"],
                "word_count": word_count,
                "page_count": doc.get("page_count", 0),
                "ocr_quality": doc.get("ocr_quality"),
                "full_doc": {
                    "coverage": full_doc_coverage,
                    "truncated": word_count > 8192,
                    "words_lost": max(0, word_count - 8192),
                },
                "passage_chunking": {
                    "n_chunks": chunk_analysis["n_chunks"],
                    "coverage_at_k": chunk_analysis["coverage_at_k"],
                    "mean_inter_chunk_similarity": mean_inter_chunk_sim,
                    "mean_chunk_to_doc_similarity": mean_chunk_doc_sim,
                    "max_chunk_to_doc_similarity": max_chunk_doc_sim,
                },
            }
            percentile_results.append(doc_result)
        
        # Aggregate
        if percentile_results:
            results[p_label] = {
                "n_documents": len(percentile_results),
                "mean_word_count": np.mean([r["word_count"] for r in percentile_results]),
                "full_doc_mean_coverage": np.mean([r["full_doc"]["coverage"] for r in percentile_results]),
                "pct_truncated": np.mean([r["full_doc"]["truncated"] for r in percentile_results]),
                "mean_n_chunks": np.mean([r["passage_chunking"]["n_chunks"] for r in percentile_results]),
                "mean_coverage_at_5": np.mean([r["passage_chunking"]["coverage_at_k"][5] for r in percentile_results]),
                "mean_inter_chunk_sim": np.mean([r["passage_chunking"]["mean_inter_chunk_similarity"] for r in percentile_results]),
                "mean_chunk_doc_sim": np.mean([r["passage_chunking"]["mean_chunk_to_doc_similarity"] for r in percentile_results]),
                "detailed": percentile_results,
            }
    
    return results


# ═══════════════════════════════════════════════════════
# Main Experiment Runner
# ═══════════════════════════════════════════════════════

def run_experiment(corpus_path: Path, output_dir: Path):
    """Run the document length experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading corpus...")
    records = load_corpus(corpus_path)
    print(f"Loaded {len(records)} documents")
    
    # ─── Length Distribution ───
    print("\nComputing length distribution...")
    distribution = compute_length_percentiles(records)
    
    print(f"\nDocument Length Distribution:")
    print(f"  Mean:    {distribution['stats']['mean']:,.0f} words")
    print(f"  Median:  {distribution['stats']['median']:,.0f} words")
    print(f"  Min:     {distribution['stats']['min']:,} words")
    print(f"  Max:     {distribution['stats']['max']:,} words")
    print(f"\n  Percentiles:")
    for label, info in distribution["percentiles"].items():
        print(f"    {label}: {info['word_count']:,} words")
    
    # ─── Sample by Percentile ───
    percentile_samples = sample_by_percentile(
        records, target_percentiles=[25, 50, 90, 99], n_per_percentile=5
    )
    
    # ─── Compare Retrieval Strategies ───
    print("\nComparing retrieval strategies by document length...")
    strategy_results = compare_retrieval_strategies(
        records, percentile_samples, output_dir
    )
    
    # ─── Save Results ───
    experiment_results = {
        "experiment": "Experiment 3: Document Length Challenge",
        "corpus_size": len(records),
        "distribution": {
            "stats": distribution["stats"],
            "percentiles": {
                k: {pk: pv for pk, pv in v.items() if pk != "doc_id"}
                for k, v in distribution["percentiles"].items()
            },
        },
        "strategy_comparison": {
            k: {sk: sv for sk, sv in v.items() if sk != "detailed"}
            for k, v in strategy_results.items()
        },
    }
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    experiment_results = convert(experiment_results)
    
    with open(output_dir / "experiment3_results.json", "w") as f:
        json.dump(experiment_results, f, indent=2)
    
    # Save detailed results
    strategy_results_serializable = convert(strategy_results)
    with open(output_dir / "experiment3_detailed.json", "w") as f:
        json.dump(strategy_results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print_length_summary(experiment_results)
    
    return experiment_results


def print_length_summary(results: dict):
    """Print summary table."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 3 RESULTS: Document Length Challenge")
    print(f"{'='*80}")
    
    comp = results.get("strategy_comparison", {})
    
    print(f"\n{'Percentile':<12} {'Mean Words':>12} {'Chunks':>10} "
          f"{'Cov@5':>10} {'Trunc%':>10} {'Inter-sim':>10}")
    print(f"{'─'*80}")
    
    for p_label in ["P25", "P50", "P90", "P99"]:
        if p_label not in comp:
            continue
        c = comp[p_label]
        print(f"{p_label:<12} "
              f"{c.get('mean_word_count', 0):>12,.0f} "
              f"{c.get('mean_n_chunks', 0):>10.0f} "
              f"{c.get('mean_coverage_at_5', 0)*100:>9.1f}% "
              f"{c.get('pct_truncated', 0)*100:>9.1f}% "
              f"{c.get('mean_inter_chunk_sim', 0):>10.3f}")
    
    # Key finding
    if "P99" in comp and "P25" in comp:
        p99_cov = comp["P99"].get("mean_coverage_at_5", 1)
        p25_cov = comp["P25"].get("mean_coverage_at_5", 1)
        if p25_cov > 0:
            coverage_drop = (1 - p99_cov / p25_cov) * 100
            print(f"\n  ⚠ Coverage@5 drops {coverage_drop:.0f}% for 99th vs 25th percentile documents")
            print(f"    → Standard chunking recovers only {p99_cov*100:.1f}% of content for longest docs")
    
    print(f"{'='*80}")


def dry_run():
    """Validate experiment setup."""
    print("Experiment 3: Dry Run Validation")
    print("="*60)
    
    checks = []
    try:
        from sentence_transformers import SentenceTransformer
        checks.append(("sentence_transformers", "✓"))
    except ImportError:
        checks.append(("sentence_transformers", "✗"))
    
    try:
        import numpy as np
        checks.append(("numpy", "✓"))
    except ImportError:
        checks.append(("numpy", "✗"))
    
    for name, status in checks:
        print(f"  {name}: {status}")
    
    return all("✓" in c[1] for c in checks)


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Document Length")
    parser.add_argument("--corpus", type=str, default="data/corpus.jsonl")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    if args.dry_run:
        dry_run()
        return
    
    run_experiment(Path(args.corpus), Path(args.output))


if __name__ == "__main__":
    main()
