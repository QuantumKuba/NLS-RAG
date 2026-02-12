#!/usr/bin/env python3
"""
Master orchestrator: runs the full NLS evaluation pipeline end-to-end.

Steps:
  1. Download & extract data (if needed)
  2. Build corpus
  3. Generate queries
  4. Run Experiment 1 (Retrieval)
  5. Run Experiment 2 (RAG)
  6. Run Experiment 3 (Length)
  7. Run Experiment 4 (Temporal Analysis)
  8. Run Experiment 5 (Cross-Collection)
  9. Run Experiment 6 (Noise Robustness)
  10. Run Experiment 7 (Corpus Statistics)
  11. Generate results & figures

Usage:
    python run_all.py                    # Run everything
    python run_all.py --from-step 3      # Resume from step 3 (query generation)
    python run_all.py --steps 1 2 4      # Run only specific steps
    python run_all.py --dry-run          # Validate all dependencies
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


STEPS = {
    1: {
        "name": "Download & Extract Data",
        "script": "scripts/download_data.py",
        "args": ["--collection", "indiaraj", "--output-dir", str(DATA_DIR)],
        "description": "Downloads NLS data from HuggingFace and extracts archives.",
    },
    2: {
        "name": "Build Corpus",
        "script": "scripts/build_corpus.py",
        "args": [
            "--data-dir", str(DATA_DIR / "extracted"),
            "--output", str(DATA_DIR / "corpus.jsonl"),
            "--sample", "5000",
        ],
        "description": "Parses ALTO/METS files, builds JSONL corpus with OCR quality scores.",
    },
    3: {
        "name": "Generate Queries",
        "script": "scripts/create_queries.py",
        "args": [
            "--corpus", str(DATA_DIR / "corpus.jsonl"),
            "--output", str(DATA_DIR / "queries"),
            "--n-retrieval", "50",
            "--n-rag", "30",
        ],
        "description": "Generates retrieval queries and RAG questions via LLM.",
    },
    4: {
        "name": "Experiment 1: Retrieval",
        "script": "experiments/experiment1_retrieval.py",
        "args": [
            "--corpus", str(DATA_DIR / "corpus.jsonl"),
            "--queries", str(DATA_DIR / "queries"),
            "--output", str(RESULTS_DIR),
        ],
        "description": "BM25 vs Dense retrieval, stratified by OCR quality.",
    },
    5: {
        "name": "Experiment 2: RAG",
        "script": "experiments/experiment2_rag.py",
        "args": [
            "--corpus", str(DATA_DIR / "corpus.jsonl"),
            "--questions", str(DATA_DIR / "queries" / "rag_questions.json"),
            "--output", str(RESULTS_DIR),
        ],
        "description": "RAG hallucination under archival noise.",
    },
    6: {
        "name": "Experiment 3: Length",
        "script": "experiments/experiment3_length.py",
        "args": [
            "--corpus", str(DATA_DIR / "corpus.jsonl"),
            "--output", str(RESULTS_DIR),
        ],
        "description": "Document length impact on retrieval.",
    },
    7: {
        "name": "Experiment 4: Temporal Analysis",
        "script": "experiments/experiment4_temporal.py",
        "args": [
            "--corpus", str(DATA_DIR / "corpus.jsonl"),
            "--output", str(RESULTS_DIR),
        ],
        "description": "Lexical drift and cross-temporal retrieval across centuries.",
    },
    8: {
        "name": "Experiment 5: Cross-Collection",
        "script": "experiments/experiment5_cross_collection.py",
        "args": [
            "--corpus", str(DATA_DIR / "corpus.jsonl"),
            "--output", str(RESULTS_DIR),
        ],
        "description": "OCR quality, vocabulary, and retrieval comparison across collections.",
    },
    9: {
        "name": "Experiment 6: Noise Robustness",
        "script": "experiments/experiment6_noise_robustness.py",
        "args": [
            "--corpus", str(DATA_DIR / "corpus.jsonl"),
            "--output", str(RESULTS_DIR),
        ],
        "description": "BM25 vs Dense retrieval robustness under synthetic OCR noise.",
    },
    10: {
        "name": "Experiment 7: Corpus Statistics",
        "script": "experiments/experiment7_corpus_stats.py",
        "args": [
            "--corpus", str(DATA_DIR / "corpus.jsonl"),
            "--output", str(RESULTS_DIR),
        ],
        "description": "Vocabulary richness, Zipf's law, OCR distributions, TF-IDF analysis.",
    },
    11: {
        "name": "Generate Results",
        "script": "analysis/generate_results.py",
        "args": [
            "--results-dir", str(RESULTS_DIR),
            "--output", str(FIGURES_DIR),
        ],
        "description": "Generate paper-ready figures and LaTeX tables.",
    },
}


def run_step(step_num: int, dry_run: bool = False) -> bool:
    """Run a single pipeline step."""
    step = STEPS[step_num]
    
    print(f"\n{'═'*60}")
    print(f"  Step {step_num}: {step['name']}")
    print(f"  {step['description']}")
    print(f"{'═'*60}\n")
    
    script_path = PROJECT_ROOT / step["script"]
    if not script_path.exists():
        print(f"  ✗ Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    
    if dry_run:
        cmd.append("--dry-run")
    else:
        cmd.extend(step["args"])
    
    print(f"  Running: {' '.join(cmd)}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,  # Stream output
            text=True,
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n  ✓ Step {step_num} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"\n  ✗ Step {step_num} failed (exit code {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n  ✗ Step {step_num} error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="NLS Evaluation Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1: Download & Extract Data
  2: Build Corpus (ALTO/METS → JSONL)
  3: Generate Queries (LLM-powered)
  4: Experiment 1 - Retrieval Under OCR Noise
  5: Experiment 2 - RAG Hallucination
  6: Experiment 3 - Document Length
  7: Experiment 4 - Temporal Analysis
  8: Experiment 5 - Cross-Collection
  9: Experiment 6 - Noise Robustness
  10: Experiment 7 - Corpus Statistics
  11: Generate Results (Figures + Tables)
        """,
    )
    parser.add_argument(
        "--from-step", type=int, default=1,
        help="Start from this step (default: 1)",
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", default=None,
        help="Run only these specific steps",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate dependencies without running experiments",
    )
    parser.add_argument(
        "--stop-on-error", action="store_true", default=True,
        help="Stop pipeline on first error (default: True)",
    )
    args = parser.parse_args()
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  NLS-CH-Multimodal SIGIR Evaluation Pipeline             ║")
    print("║  NeuraSearchLab/NLS-CH-Multimodal                        ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    if args.steps:
        steps_to_run = args.steps
    else:
        steps_to_run = [s for s in sorted(STEPS.keys()) if s >= args.from_step]
    
    print(f"\nSteps to run: {steps_to_run}")
    if args.dry_run:
        print("Mode: DRY RUN (dependency validation only)")
    
    results = {}
    total_start = time.time()
    
    for step_num in steps_to_run:
        if step_num not in STEPS:
            print(f"\n  ⚠ Unknown step: {step_num}")
            continue
        
        success = run_step(step_num, dry_run=args.dry_run)
        results[step_num] = success
        
        if not success and args.stop_on_error:
            print(f"\n⚠ Pipeline stopped at step {step_num}. Fix the error and resume with:")
            print(f"  python run_all.py --from-step {step_num}")
            break
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print(f"\n{'═'*60}")
    print(f"  Pipeline Summary ({total_elapsed:.0f}s total)")
    print(f"{'═'*60}")
    for step_num, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} Step {step_num}: {STEPS[step_num]['name']}")
    
    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\n  {len(failed)} step(s) failed: {failed}")
        sys.exit(1)
    else:
        print(f"\n  ✓ All steps completed successfully!")
        print(f"\n  Results: {RESULTS_DIR}")
        print(f"  Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
