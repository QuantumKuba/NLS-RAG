# NLS-CH-Multimodal SIGIR Evaluation Pipeline

Reproducible evaluation pipeline for the [NLS-CH-Multimodal](https://huggingface.co/datasets/NeuraSearchLab/NLS-CH-Multimodal) dataset, demonstrating real-world IR challenges from the National Library of Scotland's digitized archives.

## Key Experiments

| #   | Experiment                    | Research Question                                            |
| --- | ----------------------------- | ------------------------------------------------------------ |
| 1   | **Retrieval Under OCR Noise** | How much does OCR quality degrade BM25 vs dense retrieval?   |
| 2   | **RAG Hallucination**         | Do LLMs hallucinate more on low-quality OCR documents?       |
| 3   | **Document Length**           | Does standard chunking fail on extreme-length archival docs? |

## Quick Start

```bash
# 1. Setup
cp .env.example .env    # Add your OPENROUTER_API_KEY
pip install -r requirements.txt

# 2. Run full pipeline
python run_all.py

# 3. Or run individual steps
python run_all.py --steps 1 2       # Download + build corpus
python run_all.py --from-step 4     # Run experiments only
python run_all.py --dry-run         # Validate dependencies
```

## Project Structure

```
NLS-RAG/
├── run_all.py                      # Master orchestrator (7 steps)
├── scripts/
│   ├── download_data.py            # Download NLS data from HuggingFace
│   ├── parse_alto.py               # ALTO XML parser (OCR text + confidence)
│   ├── parse_mets.py               # METS XML parser (metadata extraction)
│   ├── build_corpus.py             # Build JSONL corpus with OCR quality tiers
│   └── create_queries.py           # LLM-powered query generation + verification
├── experiments/
│   ├── experiment1_retrieval.py    # BM25 vs Dense, stratified by OCR tier
│   ├── experiment2_rag.py          # RAG hallucination analysis
│   └── experiment3_length.py       # Document length impact
├── analysis/
│   └── generate_results.py         # Publication figures + LaTeX tables
├── tests/
│   └── test_parse_alto.py          # Unit tests for ALTO parser
├── data/                           # Generated data (gitignored)
├── results/                        # Experiment results (gitignored)
└── figures/                        # Publication figures (gitignored)
```

## Pipeline Steps

1. **Download & Extract** — Fetches RAR archives from HuggingFace, extracts ALTO/METS XML
2. **Build Corpus** — Parses ALTO (OCR text + word confidence) and METS (metadata), stratified sampling
3. **Generate Queries** — LLM generates 50 retrieval + 30 RAG queries via OpenRouter
4. **Experiment 1** — BM25 vs bge-small-en-v1.5 with FAISS, nDCG@10/MRR/Recall@10 by OCR tier
5. **Experiment 2** — RAG with fixed/semantic chunking, hallucination rate vs OCR quality
6. **Experiment 3** — Coverage and coherence analysis across document length percentiles
7. **Generate Results** — 5 publication figures + 2 LaTeX tables

## Requirements

- Python 3.9+
- `OPENROUTER_API_KEY` in `.env` (for query generation and RAG experiments)
- ~10GB disk space for data

## Tests

```bash
python -m pytest tests/ -v
```
# NLS-RAG
