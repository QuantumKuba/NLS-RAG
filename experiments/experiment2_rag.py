"""
Experiment 2: RAG Hallucination Under Archival Noise.

Tests RAG pipeline on NLS documents to measure how OCR noise affects
LLM answer quality, hallucination rates, and confidence calibration.

Key metrics: Factuality (%), hallucination rate vs OCR accuracy,
confidence-hallucination gap.

Uses OpenRouter for LLM inference.

Usage:
    python experiments/experiment2_rag.py --corpus data/corpus.jsonl --questions data/queries/rag_questions.json
    python experiments/experiment2_rag.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

import numpy as np

load_dotenv()

# Prevent tokenizer parallelism segfaults on ARM Mac / Python 3.9
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

# OpenRouter setup
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models to evaluate
DEFAULT_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct",
]


# ═══════════════════════════════════════════════════════
# Chunking Strategies
# ═══════════════════════════════════════════════════════

def chunk_fixed(text: str, chunk_size: int = 512, overlap: int = 50) -> list[dict]:
    """Fixed-size token chunking with overlap."""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "text": chunk_text,
            "start_word": start,
            "end_word": end,
            "strategy": "fixed_512",
        })
        start += chunk_size - overlap
    
    return chunks


def chunk_semantic(text: str, max_chunk_size: int = 512) -> list[dict]:
    """Semantic chunking based on paragraph/section boundaries.
    
    Splits on paragraph breaks, then merges small paragraphs up to max_chunk_size.
    Better for preserving document structure from OCR'd archival text.
    """
    # Split on double newlines (paragraph boundaries)
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_words = len(para.split())
        
        if current_size + para_words > max_chunk_size and current_chunk:
            chunks.append({
                "text": "\n\n".join(current_chunk),
                "n_paragraphs": len(current_chunk),
                "strategy": "semantic",
            })
            current_chunk = []
            current_size = 0
        
        current_chunk.append(para)
        current_size += para_words
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append({
            "text": "\n\n".join(current_chunk),
            "n_paragraphs": len(current_chunk),
            "strategy": "semantic",
        })
    
    return chunks


# ═══════════════════════════════════════════════════════
# RAG Pipeline
# ═══════════════════════════════════════════════════════

def build_chunk_index(corpus: dict, strategy: str = "fixed") -> dict:
    """Build a simple chunk index from the corpus."""
    chunk_fn = chunk_fixed if strategy == "fixed" else chunk_semantic
    
    all_chunks = []
    for doc_id, record in corpus.items():
        text = record.get("text", "")
        if not text.strip():
            continue
        
        doc_chunks = chunk_fn(text)
        for i, chunk in enumerate(doc_chunks):
            chunk["doc_id"] = doc_id
            chunk["chunk_id"] = f"{doc_id}_chunk_{i}"
            chunk["ocr_quality"] = record.get("ocr_quality")
            chunk["ocr_quality_tier"] = record.get("ocr_quality_tier")
            all_chunks.append(chunk)
    
    return all_chunks


def retrieve_chunks(query: str, chunks: list[dict], embeddings: np.ndarray,
                    model, k: int = 5) -> list[dict]:
    """Retrieve top-k relevant chunks using dense retrieval."""
    query_emb = model.encode([query], normalize_embeddings=True)
    
    scores = np.dot(embeddings, query_emb.T).squeeze()
    top_indices = np.argsort(scores)[-k:][::-1]
    
    results = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk["retrieval_score"] = float(scores[idx])
        results.append(chunk)
    
    return results


def query_llm(question: str, context_chunks: list[dict], model_name: str,
              client) -> dict:
    """Query LLM with retrieved context and get answer + confidence."""
    # Build context from chunks
    context = "\n\n---\n\n".join([c["text"] for c in context_chunks])
    
    prompt = f"""You are a historical research assistant analyzing digitized archival documents.
The following passages are from historical documents that may contain OCR errors 
(e.g., character substitutions, missing words, garbled text).

CONTEXT:
{context}

QUESTION: {question}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information, say "INSUFFICIENT_CONTEXT"
3. Be aware that OCR errors may affect readability
4. Rate your confidence in the answer from 0 to 100

Respond in this exact JSON format:
{{
    "answer": "your answer here",
    "confidence": 85,
    "reasoning": "brief explanation of how you derived the answer",
    "ocr_issues_noted": "any OCR errors you noticed that affected comprehension"
}}"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temp for factual answers
            max_tokens=500,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract key fields with regex
            answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', content)
            conf_match = re.search(r'"confidence"\s*:\s*(\d+)', content)
            result = {
                "answer": answer_match.group(1) if answer_match else content[:200],
                "confidence": int(conf_match.group(1)) if conf_match else 50,
                "reasoning": "Failed to parse structured response",
                "ocr_issues_noted": "",
            }
        
        result["model"] = model_name
        result["raw_response"] = response.choices[0].message.content
        return result
        
    except Exception as e:
        return {
            "answer": f"ERROR: {str(e)}",
            "confidence": 0,
            "reasoning": "",
            "ocr_issues_noted": "",
            "model": model_name,
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════

def evaluate_answer(predicted: str, ground_truth: str) -> dict:
    """Evaluate a predicted answer against ground truth."""
    pred_lower = predicted.lower().strip()
    gt_lower = ground_truth.lower().strip()
    
    # Exact match
    exact_match = pred_lower == gt_lower
    
    # Contains match (ground truth appears in prediction or vice versa)
    contains_match = gt_lower in pred_lower or pred_lower in gt_lower
    
    # Token overlap (F1-like)
    pred_tokens = set(pred_lower.split())
    gt_tokens = set(gt_lower.split())
    
    if pred_tokens and gt_tokens:
        overlap = pred_tokens & gt_tokens
        precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
        recall = len(overlap) / len(gt_tokens) if gt_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        f1 = 0.0
        precision = 0.0
        recall = 0.0
    
    # Check for insufficient context response
    is_abstained = "insufficient_context" in pred_lower or "not enough" in pred_lower
    
    # Hallucination: confident answer that doesn't match ground truth
    is_hallucination = not contains_match and not is_abstained and f1 < 0.3
    
    return {
        "exact_match": exact_match,
        "contains_match": contains_match,
        "token_f1": f1,
        "token_precision": precision,
        "token_recall": recall,
        "is_abstained": is_abstained,
        "is_hallucination": is_hallucination,
    }


def compute_rag_metrics(results: list[dict]) -> dict:
    """Compute aggregate RAG metrics."""
    if not results:
        return {}
    
    metrics = {
        "n_questions": len(results),
        "exact_match_rate": np.mean([r["eval"]["exact_match"] for r in results]),
        "contains_match_rate": np.mean([r["eval"]["contains_match"] for r in results]),
        "mean_token_f1": np.mean([r["eval"]["token_f1"] for r in results]),
        "hallucination_rate": np.mean([r["eval"]["is_hallucination"] for r in results]),
        "abstention_rate": np.mean([r["eval"]["is_abstained"] for r in results]),
        "mean_confidence": np.mean([r["llm_response"]["confidence"] for r in results]),
    }
    
    # Confidence-hallucination gap
    hallucinated = [r for r in results if r["eval"]["is_hallucination"]]
    correct = [r for r in results if r["eval"]["contains_match"]]
    
    if hallucinated:
        metrics["hallucinated_mean_confidence"] = np.mean(
            [r["llm_response"]["confidence"] for r in hallucinated]
        )
    if correct:
        metrics["correct_mean_confidence"] = np.mean(
            [r["llm_response"]["confidence"] for r in correct]
        )
    
    # Confidence-hallucination gap
    if hallucinated and correct:
        metrics["confidence_hallucination_gap"] = (
            metrics["correct_mean_confidence"] - metrics["hallucinated_mean_confidence"]
        )
    
    # Convert numpy types to Python natives for JSON serialization
    return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
            for k, v in metrics.items()}


def stratify_by_ocr(results: list[dict]) -> dict:
    """Stratify RAG results by OCR quality of source document."""
    by_tier = defaultdict(list)
    
    for r in results:
        tier = r.get("ocr_quality_tier", "unknown")
        by_tier[tier].append(r)
    
    stratified = {}
    for tier, tier_results in by_tier.items():
        stratified[tier] = compute_rag_metrics(tier_results)
    
    return stratified


# ═══════════════════════════════════════════════════════
# Main Experiment Runner
# ═══════════════════════════════════════════════════════

def run_experiment(corpus_path: Path, questions_path: Path, output_dir: Path,
                   models: list[str] = None, chunk_strategies: list[str] = None):
    """Run the full RAG hallucination experiment."""
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = models or DEFAULT_MODELS
    chunk_strategies = chunk_strategies or ["fixed", "semantic"]
    
    # Load data
    print("Loading corpus...")
    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            corpus[record["doc_id"]] = record
    
    print("Loading RAG questions...")
    with open(questions_path, "r") as f:
        questions = json.load(f)
    
    print(f"Corpus: {len(corpus)} documents")
    print(f"Questions: {len(questions)}")
    
    # Setup embedding model for retrieval
    print("Loading embedding model...")
    emb_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    # Setup OpenRouter client
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )
    
    all_experiment_results = {}
    
    for strategy in chunk_strategies:
        print(f"\n{'='*60}")
        print(f"Chunking Strategy: {strategy}")
        print(f"{'='*60}")
        
        # Build chunk index
        print("Building chunk index...")
        chunks = build_chunk_index(corpus, strategy=strategy)
        print(f"  Total chunks: {len(chunks)}")
        
        # Encode chunks
        print("Encoding chunks...")
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = emb_model.encode(
            chunk_texts, batch_size=64, show_progress_bar=True,
            normalize_embeddings=True,
        )
        
        for model_name in models:
            print(f"\n  Model: {model_name}")
            print(f"  {'─'*50}")
            
            results = []
            
            for i, q in enumerate(questions):
                # Find the source document
                doc_id = q.get("doc_id")
                
                # Retrieve relevant chunks
                retrieved = retrieve_chunks(
                    q["question"], chunks, chunk_embeddings, emb_model, k=5
                )
                
                # Query LLM
                llm_response = query_llm(q["question"], retrieved, model_name, client)
                
                # Evaluate
                eval_result = evaluate_answer(
                    llm_response.get("answer", ""),
                    q.get("answer", ""),
                )
                
                result = {
                    "question_id": q.get("question_id"),
                    "question": q["question"],
                    "ground_truth": q.get("answer", ""),
                    "doc_id": doc_id,
                    "ocr_quality": q.get("ocr_quality"),
                    "ocr_quality_tier": q.get("ocr_quality_tier", "unknown"),
                    "llm_response": llm_response,
                    "eval": eval_result,
                    "retrieved_chunks": [
                        {"chunk_id": c["chunk_id"], "score": c["retrieval_score"],
                         "text_preview": c["text"][:200]}
                        for c in retrieved
                    ],
                    "chunking_strategy": strategy,
                    "model": model_name,
                }
                results.append(result)
                
                if (i + 1) % 5 == 0:
                    print(f"    Processed {i+1}/{len(questions)} questions")
                
                # Rate limiting
                time.sleep(0.5)
            
            # Compute metrics
            overall_metrics = compute_rag_metrics(results)
            stratified_metrics = stratify_by_ocr(results)
            
            key = f"{strategy}_{model_name.split('/')[-1]}"
            all_experiment_results[key] = {
                "model": model_name,
                "chunking_strategy": strategy,
                "overall": overall_metrics,
                "by_ocr_tier": stratified_metrics,
                "detailed_results": results,
            }
            
            # Print summary
            print(f"\n    Results for {model_name.split('/')[-1]} ({strategy}):")
            print(f"      Hallucination rate: {overall_metrics.get('hallucination_rate', 0):.1%}")
            print(f"      Contains match:     {overall_metrics.get('contains_match_rate', 0):.1%}")
            print(f"      Mean confidence:    {overall_metrics.get('mean_confidence', 0):.1f}")
            
            if "hallucinated_mean_confidence" in overall_metrics:
                print(f"      Halluci. confidence: {overall_metrics['hallucinated_mean_confidence']:.1f}")
            if "confidence_hallucination_gap" in overall_metrics:
                print(f"      Conf-Halluci. gap:  {overall_metrics['confidence_hallucination_gap']:.1f}")
    
    # Save all results
    # Remove raw responses for the summary file (keep detailed in separate file)
    summary = {}
    for key, val in all_experiment_results.items():
        summary[key] = {k: v for k, v in val.items() if k != "detailed_results"}
    
    with open(output_dir / "experiment2_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / "experiment2_detailed.json", "w") as f:
        json.dump(all_experiment_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_dir}")
    print_rag_summary(all_experiment_results)
    
    return all_experiment_results


def print_rag_summary(results: dict):
    """Print a summary table of RAG experiment results."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 2 RESULTS: RAG Hallucination Under Archival Noise")
    print(f"{'='*80}")
    
    print(f"\n{'Config':<35} {'Halluci%':>10} {'Match%':>10} {'F1':>8} {'Conf':>8}")
    print(f"{'─'*80}")
    
    for key, val in results.items():
        m = val["overall"]
        print(f"{key:<35} "
              f"{m.get('hallucination_rate', 0)*100:>9.1f}% "
              f"{m.get('contains_match_rate', 0)*100:>9.1f}% "
              f"{m.get('mean_token_f1', 0):>8.3f} "
              f"{m.get('mean_confidence', 0):>7.1f}")
    
    # OCR tier breakdown
    print(f"\n{'By OCR Tier (Hallucination Rate)':^80}")
    print(f"{'─'*80}")
    print(f"{'Config':<35} {'High':>10} {'Medium':>10} {'Low':>10}")
    print(f"{'─'*80}")
    
    for key, val in results.items():
        by_tier = val.get("by_ocr_tier", {})
        high = by_tier.get("high", {}).get("hallucination_rate", "-")
        med = by_tier.get("medium", {}).get("hallucination_rate", "-")
        low = by_tier.get("low", {}).get("hallucination_rate", "-")
        
        def fmt(v):
            return f"{v*100:.1f}%" if isinstance(v, (int, float)) else v
        
        print(f"{key:<35} {fmt(high):>10} {fmt(med):>10} {fmt(low):>10}")
    
    print(f"{'='*80}")


def dry_run():
    """Validate experiment setup."""
    print("Experiment 2: Dry Run Validation")
    print("="*60)
    
    checks = []
    
    try:
        from openai import OpenAI
        checks.append(("openai", "✓"))
    except ImportError:
        checks.append(("openai", "✗ pip install openai"))
    
    try:
        from sentence_transformers import SentenceTransformer
        checks.append(("sentence_transformers", "✓"))
    except ImportError:
        checks.append(("sentence_transformers", "✗ pip install sentence-transformers"))
    
    if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_key_here":
        checks.append(("OPENROUTER_API_KEY", "✓"))
    else:
        checks.append(("OPENROUTER_API_KEY", "✗ Set in .env file"))
    
    try:
        from dotenv import load_dotenv
        checks.append(("python-dotenv", "✓"))
    except ImportError:
        checks.append(("python-dotenv", "✗ pip install python-dotenv"))
    
    for name, status in checks:
        print(f"  {name}: {status}")
    
    return all("✓" in c[1] for c in checks)


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: RAG Hallucination")
    parser.add_argument("--corpus", type=str, default="data/corpus.jsonl")
    parser.add_argument("--questions", type=str, default="data/queries/rag_questions.json")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                       help="LLM models to evaluate (OpenRouter names)")
    parser.add_argument("--strategies", type=str, nargs="+", default=None,
                       choices=["fixed", "semantic"],
                       help="Chunking strategies to compare")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    if args.dry_run:
        dry_run()
        return
    
    run_experiment(
        Path(args.corpus), Path(args.questions), Path(args.output),
        models=args.models, chunk_strategies=args.strategies,
    )


if __name__ == "__main__":
    main()
