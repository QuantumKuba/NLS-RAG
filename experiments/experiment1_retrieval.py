"""
Experiment 1: Sparse vs Dense Retrieval Under OCR Noise.

Evaluates BM25 and dense retrieval (bge-small-en-v1.5) on the NLS corpus,
stratifying results by OCR quality tiers to demonstrate retrieval degradation
under archival noise.

Key metrics: nDCG@10, MRR, Recall@10 — stratified by OCR quality tier.

Usage:
    python experiments/experiment1_retrieval.py --corpus data/corpus.jsonl --queries data/queries/
    python experiments/experiment1_retrieval.py --dry-run  # Validate setup without running
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import numpy as np
from collections import defaultdict
from pathlib import Path

# Prevent tokenizer parallelism segfaults on ARM Mac / Python 3.9
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_corpus(corpus_path: Path) -> dict:
    """Load corpus as {doc_id: record}."""
    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            corpus[record["doc_id"]] = record
    return corpus


def load_queries(query_dir: Path) -> tuple[dict, dict]:
    """Load queries and qrels in BEIR format."""
    queries = {}
    with open(query_dir / "queries.jsonl", "r") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    
    qrels = defaultdict(dict)
    with open(query_dir / "qrels.tsv", "r") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qrels[parts[0]][parts[1]] = int(parts[2])
    
    return queries, dict(qrels)


def load_query_metadata(query_dir: Path) -> dict:
    """Load full query metadata for stratification."""
    full_path = query_dir / "queries_full.json"
    if full_path.exists():
        with open(full_path, "r") as f:
            queries_full = json.load(f)
        return {q["query_id"]: q for q in queries_full}
    return {}


# ═══════════════════════════════════════════════════════
# BM25 Retrieval
# ═══════════════════════════════════════════════════════

def run_bm25(corpus: dict, queries: dict, k: int = 10) -> dict:
    """Run BM25 retrieval using rank_bm25."""
    from rank_bm25 import BM25Okapi
    
    print("Building BM25 index...")
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did].get("text", "") for did in doc_ids]
    
    # Simple tokenization
    tokenized = [text.lower().split() for text in doc_texts]
    bm25 = BM25Okapi(tokenized)
    
    print(f"  Indexed {len(doc_ids)} documents")
    
    results = {}
    for qid, query_text in queries.items():
        tokenized_query = query_text.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k
        top_indices = np.argsort(scores)[-k:][::-1]
        results[qid] = {
            doc_ids[idx]: float(scores[idx])
            for idx in top_indices
            if scores[idx] > 0
        }
    
    return results


# ═══════════════════════════════════════════════════════
# Dense Retrieval
# ═══════════════════════════════════════════════════════

def run_dense_retrieval(corpus: dict, queries: dict, k: int = 10,
                        model_name: str = "BAAI/bge-small-en-v1.5",
                        batch_size: int = 32) -> dict:
    """Run dense retrieval using sentence-transformers + FAISS.
    
    Uses FAISS IndexFlatIP for exact inner-product search.
    With normalized embeddings, inner product = cosine similarity.
    """
    import gc
    from sentence_transformers import SentenceTransformer
    import faiss
    
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did].get("text", "")[:8192] for did in doc_ids]  # Truncate for embedding
    
    print(f"Encoding {len(doc_texts)} documents...")
    start = time.time()
    doc_embeddings = model.encode(
        doc_texts, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - start
    print(f"  Encoded in {elapsed:.1f}s")
    
    # Free memory from the large document texts before query encoding
    del doc_texts
    gc.collect()
    
    # Build FAISS index
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vecs)
    index.add(doc_embeddings.astype(np.float32))
    print(f"  FAISS index built (dim={dim}, n={index.ntotal})")
    
    # Free the raw embeddings array now that they're in FAISS
    del doc_embeddings
    gc.collect()
    
    # Encode queries and search
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    print(f"Encoding {len(query_texts)} queries...")
    sys.stdout.flush()
    query_embeddings = model.encode(
        query_texts, normalize_embeddings=True,
    )
    
    print("  Searching FAISS index...")
    sys.stdout.flush()
    scores, indices = index.search(query_embeddings.astype(np.float32), k)
    
    results = {}
    for i, qid in enumerate(query_ids):
        results[qid] = {}
        for j in range(k):
            if indices[i][j] >= 0:
                did = doc_ids[indices[i][j]]
                results[qid][did] = float(scores[i][j])
    
    return results


# ═══════════════════════════════════════════════════════
# Evaluation Metrics
# ═══════════════════════════════════════════════════════

def dcg_at_k(ranked_list: list[float], k: int) -> float:
    """Compute DCG@k."""
    dcg = 0.0
    for i, rel in enumerate(ranked_list[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(results: dict, qrels: dict, k: int = 10) -> dict:
    """Compute nDCG@k per query."""
    scores = {}
    for qid in qrels:
        if qid not in results:
            scores[qid] = 0.0
            continue
        
        # Get relevance of retrieved documents
        ranked = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
        ranked_rels = [float(qrels[qid].get(did, 0)) for did, _ in ranked]
        
        # Ideal ranking
        ideal_rels = sorted(qrels[qid].values(), reverse=True)[:k]
        ideal_rels = [float(r) for r in ideal_rels]
        
        dcg = dcg_at_k(ranked_rels, k)
        idcg = dcg_at_k(ideal_rels, k)
        
        scores[qid] = dcg / idcg if idcg > 0 else 0.0
    
    return scores


def mrr(results: dict, qrels: dict) -> dict:
    """Compute Mean Reciprocal Rank per query."""
    scores = {}
    for qid in qrels:
        if qid not in results:
            scores[qid] = 0.0
            continue
        
        ranked = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
        for rank, (did, _) in enumerate(ranked, 1):
            if qrels[qid].get(did, 0) > 0:
                scores[qid] = 1.0 / rank
                break
        else:
            scores[qid] = 0.0
    
    return scores


def recall_at_k(results: dict, qrels: dict, k: int = 10) -> dict:
    """Compute Recall@k per query."""
    scores = {}
    for qid in qrels:
        if qid not in results:
            scores[qid] = 0.0
            continue
        
        ranked = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
        retrieved_relevant = sum(1 for did, _ in ranked if qrels[qid].get(did, 0) > 0)
        total_relevant = sum(1 for v in qrels[qid].values() if v > 0)
        
        scores[qid] = retrieved_relevant / total_relevant if total_relevant > 0 else 0.0
    
    return scores


# ═══════════════════════════════════════════════════════
# Stratified Analysis
# ═══════════════════════════════════════════════════════

def stratify_by_ocr_tier(query_scores: dict, query_metadata: dict) -> dict:
    """Group query scores by OCR quality tier of the relevant document."""
    by_tier = defaultdict(list)
    
    for qid, score in query_scores.items():
        meta = query_metadata.get(qid, {})
        tier = meta.get("ocr_quality_tier", "unknown")
        by_tier[tier].append(score)
    
    return dict(by_tier)


def compute_stratified_metrics(results: dict, qrels: dict, 
                                 query_metadata: dict) -> dict:
    """Compute all metrics stratified by OCR quality tier."""
    ndcg_scores = ndcg_at_k(results, qrels, k=10)
    mrr_scores = mrr(results, qrels)
    recall_scores = recall_at_k(results, qrels, k=10)
    
    ndcg_by_tier = stratify_by_ocr_tier(ndcg_scores, query_metadata)
    mrr_by_tier = stratify_by_ocr_tier(mrr_scores, query_metadata)
    recall_by_tier = stratify_by_ocr_tier(recall_scores, query_metadata)
    
    report = {
        "overall": {
            "nDCG@10": {
                "mean": float(np.mean(list(ndcg_scores.values()))),
                "std": float(np.std(list(ndcg_scores.values()))),
            },
            "MRR": {
                "mean": float(np.mean(list(mrr_scores.values()))),
                "std": float(np.std(list(mrr_scores.values()))),
            },
            "Recall@10": {
                "mean": float(np.mean(list(recall_scores.values()))),
                "std": float(np.std(list(recall_scores.values()))),
            },
        },
        "by_ocr_tier": {},
    }
    
    for tier in ["high", "medium", "low", "unknown"]:
        if tier in ndcg_by_tier:
            report["by_ocr_tier"][tier] = {
                "n_queries": len(ndcg_by_tier[tier]),
                "nDCG@10": {
                    "mean": float(np.mean(ndcg_by_tier[tier])),
                    "std": float(np.std(ndcg_by_tier[tier])),
                },
                "MRR": {
                    "mean": float(np.mean(mrr_by_tier.get(tier, [0]))),
                    "std": float(np.std(mrr_by_tier.get(tier, [0]))),
                },
                "Recall@10": {
                    "mean": float(np.mean(recall_by_tier.get(tier, [0]))),
                    "std": float(np.std(recall_by_tier.get(tier, [0]))),
                },
            }
    
    return report


def find_qualitative_examples(results: dict, qrels: dict, corpus: dict,
                                query_metadata: dict, queries: dict,
                                n_examples: int = 5) -> list[dict]:
    """Find interesting failure cases for qualitative analysis."""
    examples = []
    
    # Find queries where BM25 failed (score = 0) on low-OCR docs
    for qid in qrels:
        meta = query_metadata.get(qid, {})
        tier = meta.get("ocr_quality_tier", "unknown")
        
        if tier != "low":
            continue
        
        # Check if relevant doc was retrieved
        relevant_docs = [did for did, rel in qrels[qid].items() if rel > 0]
        ranked = sorted(results.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
        
        retrieved_ids = [did for did, _ in ranked[:10]]
        missed = [did for did in relevant_docs if did not in retrieved_ids]
        
        if missed:
            doc = corpus.get(missed[0], {})
            text_snippet = doc.get("text", "")[:300]
            
            examples.append({
                "query_id": qid,
                "query": queries.get(qid, ""),
                "relevant_doc_id": missed[0],
                "ocr_quality": meta.get("ocr_quality"),
                "ocr_tier": tier,
                "text_snippet": text_snippet,
                "failure_type": "relevant_doc_not_retrieved",
                "explanation": f"OCR quality {meta.get('ocr_quality', 'N/A'):.2f} - "
                             f"document likely contains OCR errors preventing match",
            })
        
        if len(examples) >= n_examples:
            break
    
    return examples


# ═══════════════════════════════════════════════════════
# Statistical Tests
# ═══════════════════════════════════════════════════════

def significance_test(scores_a: list[float], scores_b: list[float]) -> dict:
    """Paired statistical test between two systems."""
    from scipy import stats
    
    if len(scores_a) != len(scores_b) or len(scores_a) < 3:
        return {"test": "insufficient_data"}
    
    # Paired t-test
    t_stat, p_val_t = stats.ttest_rel(scores_a, scores_b)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        w_stat, p_val_w = stats.wilcoxon(scores_a, scores_b)
    except ValueError:
        w_stat, p_val_w = None, None
    
    return {
        "paired_ttest": {"t_statistic": float(t_stat), "p_value": float(p_val_t)},
        "wilcoxon": {
            "w_statistic": float(w_stat) if w_stat else None,
            "p_value": float(p_val_w) if p_val_w else None,
        },
        "mean_diff": float(np.mean(scores_a) - np.mean(scores_b)),
    }


# ═══════════════════════════════════════════════════════
# Main Experiment Runner
# ═══════════════════════════════════════════════════════

def run_experiment(corpus_path: Path, query_dir: Path, output_dir: Path,
                   dense_model: str = "BAAI/bge-small-en-v1.5"):
    """Run the full Experiment 1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    corpus = load_corpus(corpus_path)
    queries, qrels = load_queries(query_dir)
    query_metadata = load_query_metadata(query_dir)
    
    print(f"Corpus: {len(corpus)} documents")
    print(f"Queries: {len(queries)}")
    print(f"Qrels: {len(qrels)} queries with relevance judgments")
    
    # ─── Run BM25 ───
    print("\n" + "="*60)
    print("Running BM25 retrieval...")
    print("="*60)
    results_bm25 = run_bm25(corpus, queries, k=10)
    
    # ─── Run Dense Retrieval ───
    print("\n" + "="*60)
    print(f"Running Dense retrieval ({dense_model})...")
    print("="*60)
    results_dense = run_dense_retrieval(corpus, queries, k=10, model_name=dense_model)
    
    # ─── Evaluate ───
    print("\n" + "="*60)
    print("Computing metrics...")
    print("="*60)
    
    bm25_metrics = compute_stratified_metrics(results_bm25, qrels, query_metadata)
    dense_metrics = compute_stratified_metrics(results_dense, qrels, query_metadata)
    
    # ─── Statistical significance ───
    ndcg_bm25 = ndcg_at_k(results_bm25, qrels, k=10)
    ndcg_dense = ndcg_at_k(results_dense, qrels, k=10)
    
    common_qids = sorted(set(ndcg_bm25.keys()) & set(ndcg_dense.keys()))
    sig_test = significance_test(
        [ndcg_bm25[q] for q in common_qids],
        [ndcg_dense[q] for q in common_qids],
    )
    
    # ─── Qualitative examples ───
    bm25_examples = find_qualitative_examples(
        results_bm25, qrels, corpus, query_metadata, queries
    )
    dense_examples = find_qualitative_examples(
        results_dense, qrels, corpus, query_metadata, queries
    )
    
    # ─── Save results ───
    experiment_results = {
        "experiment": "Experiment 1: Sparse vs Dense Retrieval Under OCR Noise",
        "corpus_size": len(corpus),
        "n_queries": len(queries),
        "models": {
            "bm25": bm25_metrics,
            "dense": {
                "model_name": dense_model,
                **dense_metrics,
            },
        },
        "significance_test": sig_test,
        "qualitative_examples": {
            "bm25_failures": bm25_examples,
            "dense_failures": dense_examples,
        },
    }
    
    with open(output_dir / "experiment1_results.json", "w") as f:
        json.dump(experiment_results, f, indent=2)
    
    # Save raw results for further analysis
    with open(output_dir / "experiment1_bm25_results.json", "w") as f:
        json.dump(results_bm25, f, indent=2)
    
    with open(output_dir / "experiment1_dense_results.json", "w") as f:
        json.dump(results_dense, f, indent=2)
    
    # ─── Print summary ───
    print_results_table(bm25_metrics, dense_metrics, dense_model)
    
    print(f"\nResults saved to: {output_dir}")
    return experiment_results


def print_results_table(bm25: dict, dense: dict, dense_model: str):
    """Print a formatted results table."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1 RESULTS: Sparse vs Dense Retrieval Under OCR Noise")
    print(f"{'='*80}")
    
    # Overall
    print(f"\n{'Overall':^80}")
    print(f"{'-'*80}")
    print(f"{'Metric':<15} {'BM25':>20} {'Dense (' + dense_model.split('/')[-1] + ')':>40}")
    print(f"{'-'*80}")
    
    for metric in ["nDCG@10", "MRR", "Recall@10"]:
        bm25_val = bm25["overall"][metric]
        dense_val = dense["overall"][metric]
        print(f"{metric:<15} {bm25_val['mean']:.4f} ± {bm25_val['std']:.4f}"
              f"          {dense_val['mean']:.4f} ± {dense_val['std']:.4f}")
    
    # By OCR Tier
    print(f"\n{'By OCR Quality Tier':^80}")
    print(f"{'-'*80}")
    print(f"{'Tier':<10} {'n':>5} {'BM25 nDCG@10':>18} {'Dense nDCG@10':>18} {'Δ':>10}")
    print(f"{'-'*80}")
    
    for tier in ["high", "medium", "low"]:
        bm25_tier = bm25["by_ocr_tier"].get(tier, {})
        dense_tier = dense["by_ocr_tier"].get(tier, {})
        
        if not bm25_tier:
            continue
        
        n = bm25_tier.get("n_queries", 0)
        bm25_ndcg = bm25_tier["nDCG@10"]["mean"]
        dense_ndcg = dense_tier.get("nDCG@10", {}).get("mean", 0)
        delta = bm25_ndcg - dense_ndcg
        
        print(f"{tier:<10} {n:>5} {bm25_ndcg:>18.4f} {dense_ndcg:>18.4f} {delta:>+10.4f}")
    
    # Degradation summary
    tiers = list(bm25["by_ocr_tier"].keys())
    if "high" in tiers and "low" in tiers:
        high_ndcg = bm25["by_ocr_tier"]["high"]["nDCG@10"]["mean"]
        low_ndcg = bm25["by_ocr_tier"]["low"]["nDCG@10"]["mean"]
        drop = (high_ndcg - low_ndcg) / high_ndcg * 100 if high_ndcg > 0 else 0
        print(f"\n  ⚠ BM25 nDCG@10 drops {drop:.1f}% between high and low OCR tiers")
    
    print(f"{'='*80}")


def dry_run():
    """Validate experiment setup without running."""
    print("Experiment 1: Dry Run Validation")
    print("="*60)
    
    checks = []
    
    # Check dependencies
    try:
        import rank_bm25
        checks.append(("rank_bm25", "✓"))
    except ImportError:
        checks.append(("rank_bm25", "✗ pip install rank_bm25"))
    
    try:
        import sentence_transformers
        checks.append(("sentence_transformers", "✓"))
    except ImportError:
        checks.append(("sentence_transformers", "✗ pip install sentence-transformers"))
    
    try:
        import faiss
        checks.append(("faiss", "✓"))
    except ImportError:
        checks.append(("faiss", "✗ pip install faiss-cpu"))
    
    try:
        from scipy import stats
        checks.append(("scipy", "✓"))
    except ImportError:
        checks.append(("scipy", "✗ pip install scipy"))
    
    for name, status in checks:
        print(f"  {name}: {status}")
    
    failed = [c for c in checks if "✗" in c[1]]
    if failed:
        print(f"\n⚠ Install missing dependencies:")
        print(f"  pip install {' '.join(c[1].split('pip install ')[1] for c in failed)}")
        return False
    
    print("\n✓ All dependencies available")
    return True


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Retrieval Under OCR Noise")
    parser.add_argument("--corpus", type=str, default="data/corpus.jsonl")
    parser.add_argument("--queries", type=str, default="data/queries")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--dense-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    if args.dry_run:
        dry_run()
        return
    
    run_experiment(
        Path(args.corpus), Path(args.queries), Path(args.output), args.dense_model
    )


if __name__ == "__main__":
    main()
