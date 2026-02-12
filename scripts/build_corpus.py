"""
Build unified document corpus from extracted NLS TXT data.

Walks the extracted NLS data directory tree and produces a JSONL corpus file
with text, metadata extracted from the folder structure, and heuristic-based
OCR quality estimates (since ALTO XML confidence data is unavailable).

Actual data layout:
    data/extracted/
        indiaraj/                          # collection
            1767/                          # date range
                Correspondence_between.../  # document title (truncated)
                    22354695/              # numeric ID
                        22354695.pdf       # scanned PDF
                        22354695.txt       # OCR text

Usage:
    python scripts/build_corpus.py --data-dir data/extracted --output data/corpus.jsonl
    python scripts/build_corpus.py --data-dir data/extracted --output data/corpus.jsonl --sample 200
    python scripts/build_corpus.py --validate --corpus data/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import random
import statistics
from collections import Counter
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# OCR quality estimation from raw text (no ALTO confidence scores available)
# ---------------------------------------------------------------------------

def estimate_ocr_quality(text: str) -> float | None:
    """Estimate OCR quality from raw text using character-level heuristics.
    
    Returns a score between 0.0 (terrible) and 1.0 (perfect).
    
    Heuristic signals:
        1. Ratio of alphabetic + space + common punctuation chars
        2. Average word length (very short words → misrecognition fragments)
        3. Ratio of "garbage" tokens (no vowels, excessive special chars)
        4. Ratio of very short words (1-2 chars that aren't common words)
    """
    if not text or not text.strip():
        return None
    
    # Signal 1: clean character ratio
    clean_chars = sum(1 for c in text if c.isalpha() or c in " \n\t.,;:!?'-\"()[]")
    char_ratio = clean_chars / len(text) if text else 0
    
    # Signal 2 & 3: word-level analysis
    words = text.split()
    if not words:
        return None
    
    # Common short English words that are OK
    common_short = {
        "a", "i", "o", "an", "am", "as", "at", "be", "by", "do", "go",
        "he", "if", "in", "is", "it", "me", "my", "no", "of", "on", "or",
        "so", "to", "up", "us", "we", "&", "mr", "st", "dr", "ms",
    }
    
    # Count garbage tokens (no vowel, or mostly non-alpha)
    garbage_count = 0
    short_garbage = 0
    vowels = set("aeiouAEIOU")
    
    for w in words:
        stripped = w.strip(".,;:!?'-\"()[]")
        if not stripped:
            continue
        
        alpha_chars = sum(1 for c in stripped if c.isalpha())
        
        # Token with very few alpha chars relative to length
        if len(stripped) > 2 and alpha_chars / len(stripped) < 0.5:
            garbage_count += 1
        # Short token that isn't a known word
        elif len(stripped) <= 2 and stripped.lower() not in common_short:
            short_garbage += 1
        # Longer word with no vowels (likely OCR error)
        elif len(stripped) > 3 and not any(c in vowels for c in stripped):
            garbage_count += 1
    
    garbage_ratio = garbage_count / len(words) if words else 0
    short_garbage_ratio = short_garbage / len(words) if words else 0
    
    # Combine signals into a single score
    # Weights chosen empirically for 18th-19th century OCR text
    score = (
        0.50 * char_ratio +               # clean chars is a strong signal
        0.30 * (1.0 - garbage_ratio) +     # low garbage is good
        0.20 * (1.0 - short_garbage_ratio) # low short garbage is good
    )
    
    return max(0.0, min(1.0, score))


def ocr_quality_tier(score: float | None) -> str:
    """Classify OCR quality into tiers from score.
    
    Thresholds calibrated for the NLS indiaraj collection where
    scores range 0.81-0.99. Using tighter bands to create a
    balanced tier distribution suitable for stratified experiments.
    """
    if score is None:
        return "unknown"
    if score >= 0.98:
        return "high"
    elif score >= 0.95:
        return "medium"
    else:
        return "low"


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def extract_date_from_folder(folder_name: str) -> tuple[str, int | None]:
    """Extract date string and numeric year from folder name like '1767' or '1782-1790'.
    
    Returns (date_string, earliest_numeric_year).
    """
    # Match patterns like "1767", "1782-1790"
    match = re.match(r"(\d{4})(?:-(\d{4}))?", folder_name)
    if match:
        start_year = match.group(1)
        end_year = match.group(2)
        if end_year:
            return f"{start_year}-{end_year}", int(start_year)
        return start_year, int(start_year)
    return folder_name, None


def clean_title(folder_name: str) -> str:
    """Convert folder name back to a readable title.
    
    E.g. 'Correspondence_between_officials_of_the_East_India' → 
         'Correspondence between officials of the East India'
    """
    return folder_name.replace("_", " ").strip()


def discover_documents(data_dir: Path) -> list[dict]:
    """Discover documents in the NLS data directory.
    
    Expected structure:
        data_dir/
            collection/           (e.g. 'indiaraj')
                date_range/       (e.g. '1767', '1782-1790')
                    title_folder/ (e.g. 'Correspondence_between...')
                        id/       (e.g. '22354695')
                            id.txt             ← single-file document
                            id.pdf
                        OR:
                        id/                    ← multi-file document
                            19564449.txt       (letters, volumes, etc.)
                            19564450.txt
                            ...
    """
    documents = []
    
    for collection_dir in sorted(data_dir.iterdir()):
        if not collection_dir.is_dir() or collection_dir.name.startswith("."):
            continue
        
        collection = collection_dir.name
        
        for date_dir in sorted(collection_dir.iterdir()):
            if not date_dir.is_dir() or date_dir.name.startswith("."):
                continue
            
            date_str, date_numeric = extract_date_from_folder(date_dir.name)
            
            for title_dir in sorted(date_dir.iterdir()):
                if not title_dir.is_dir() or title_dir.name.startswith("."):
                    continue
                
                title = clean_title(title_dir.name)
                
                for id_dir in sorted(title_dir.iterdir()):
                    if not id_dir.is_dir() or id_dir.name.startswith("."):
                        continue
                    
                    doc_id = id_dir.name
                    txt_file = id_dir / f"{doc_id}.txt"
                    pdf_file = id_dir / f"{doc_id}.pdf"
                    
                    if txt_file.exists():
                        # Single-file document (most common)
                        documents.append({
                            "doc_id": doc_id,
                            "title": title,
                            "date": date_str,
                            "date_numeric": date_numeric,
                            "collection": collection,
                            "txt_file": txt_file,
                            "txt_files": None,
                            "pdf_file": pdf_file if pdf_file.exists() else None,
                            "source_dir": str(id_dir),
                        })
                    else:
                        # Multi-file document: collect all .txt files
                        txt_files = sorted([
                            f for f in id_dir.glob("*.txt")
                            if not f.name.startswith(".")
                        ])
                        if txt_files:
                            documents.append({
                                "doc_id": doc_id,
                                "title": title,
                                "date": date_str,
                                "date_numeric": date_numeric,
                                "collection": collection,
                                "txt_file": None,
                                "txt_files": txt_files,
                                "pdf_file": pdf_file if pdf_file.exists() else None,
                                "source_dir": str(id_dir),
                            })
    
    return documents


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

def _read_text_file(path: Path) -> str:
    """Read a text file, trying UTF-8 then Latin-1."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="replace")
        except Exception:
            return ""


def process_document(doc_info: dict) -> dict | None:
    """Process a single document: read text, estimate OCR quality."""
    
    if doc_info.get("txt_file"):
        # Single-file document
        text = _read_text_file(doc_info["txt_file"]).strip()
    elif doc_info.get("txt_files"):
        # Multi-file document: concatenate all TXT files (sorted by name)
        parts = []
        for f in doc_info["txt_files"]:
            part = _read_text_file(f).strip()
            if part:
                parts.append(part)
        text = "\n\n".join(parts)
    else:
        return None
    
    # Clean up carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Skip empty or near-empty documents (images, paintings, etc.)
    words = text.split()
    if len(words) < 10:
        return None
    
    # Estimate OCR quality
    quality = estimate_ocr_quality(text)
    tier = ocr_quality_tier(quality)
    
    # Estimate page count from PDF file size (rough: ~50KB per page for scanned docs)
    page_estimate = None
    if doc_info.get("pdf_file"):
        try:
            pdf_size = doc_info["pdf_file"].stat().st_size
            page_estimate = max(1, pdf_size // 50_000)
        except Exception:
            pass
    
    # Count source files
    n_source_files = 1
    if doc_info.get("txt_files"):
        n_source_files = len(doc_info["txt_files"])
    
    return {
        "doc_id": doc_info["doc_id"],
        "title": doc_info["title"],
        "author": "",  # Not available in folder structure
        "date": doc_info["date"],
        "date_numeric": doc_info["date_numeric"],
        "collection": doc_info["collection"],
        "language": "",  # Could be inferred but not available
        "subject": [],
        "text": text,
        "ocr_quality": quality,
        "ocr_quality_tier": tier,
        "word_count": len(words),
        "page_count": page_estimate,
        "char_count": len(text),
        "source_files": n_source_files,
        "source_dir": doc_info["source_dir"],
    }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def stratified_sample(records: list[dict], sample_size: int, seed: int = 42) -> list[dict]:
    """Sample documents stratified by OCR quality tier and temporal range."""
    random.seed(seed)
    
    tiers = {"high": [], "medium": [], "low": [], "unknown": []}
    for r in records:
        tier = r.get("ocr_quality_tier", "unknown")
        tiers[tier].append(r)
    
    total = len(records)
    sampled = []
    
    for tier, tier_records in tiers.items():
        if not tier_records:
            continue
        
        n = max(1, int(sample_size * len(tier_records) / total))
        n = min(n, len(tier_records))
        
        tier_records.sort(key=lambda r: r.get("date_numeric") or 1900)
        
        if len(tier_records) <= n:
            sampled.extend(tier_records)
        else:
            step = len(tier_records) / n
            indices = [int(i * step) for i in range(n)]
            sampled.extend([tier_records[i] for i in indices])
    
    random.shuffle(sampled)
    return sampled[:sample_size]


# ---------------------------------------------------------------------------
# Stats & validation
# ---------------------------------------------------------------------------

def print_corpus_stats(records: list[dict]):
    """Print summary statistics for the corpus."""
    print(f"\n{'='*60}")
    print(f"Corpus Statistics")
    print(f"{'='*60}")
    print(f"Total documents: {len(records):,}")
    
    # OCR quality distribution
    tier_counts = Counter(r.get("ocr_quality_tier", "unknown") for r in records)
    print(f"\nOCR Quality Tiers:")
    for tier in ["high", "medium", "low", "unknown"]:
        count = tier_counts.get(tier, 0)
        pct = count / len(records) * 100 if records else 0
        print(f"  {tier:>8}: {count:>6,} ({pct:>5.1f}%)")
    
    # OCR quality scores
    qualities = [r["ocr_quality"] for r in records if r.get("ocr_quality") is not None]
    if qualities:
        print(f"\nOCR Quality Scores (estimated):")
        print(f"  Mean:   {statistics.mean(qualities):.3f}")
        print(f"  Median: {statistics.median(qualities):.3f}")
        if len(qualities) > 1:
            print(f"  StdDev: {statistics.stdev(qualities):.3f}")
        print(f"  Min:    {min(qualities):.3f}")
        print(f"  Max:    {max(qualities):.3f}")
    
    # Document length distribution
    word_counts = [r["word_count"] for r in records]
    if word_counts:
        print(f"\nDocument Length (words):")
        print(f"  Mean:   {statistics.mean(word_counts):,.0f}")
        print(f"  Median: {statistics.median(word_counts):,.0f}")
        print(f"  Min:    {min(word_counts):,}")
        print(f"  Max:    {max(word_counts):,}")
        
        sorted_wc = sorted(word_counts)
        for p in [25, 50, 75, 90, 95, 99]:
            idx = min(int(len(sorted_wc) * p / 100), len(sorted_wc) - 1)
            print(f"  P{p}:    {sorted_wc[idx]:,}")
    
    # Temporal distribution
    years = [r["date_numeric"] for r in records if r.get("date_numeric")]
    if years:
        print(f"\nTemporal Range:")
        print(f"  Earliest: {min(years)}")
        print(f"  Latest:   {max(years)}")
        
        decades = Counter((y // 10) * 10 for y in years)
        print(f"  Decades covered: {len(decades)}")
        for decade in sorted(decades):
            print(f"    {decade}s: {decades[decade]:,}")
    
    # Collection distribution
    collections = Counter(r.get("collection", "unknown") for r in records)
    print(f"\nCollections:")
    for col, count in collections.most_common():
        print(f"  {col}: {count:,}")
    
    print(f"{'='*60}\n")


def validate_corpus(corpus_path: str | Path):
    """Validate an existing corpus file."""
    corpus_path = Path(corpus_path)
    
    if not corpus_path.exists():
        print(f"Corpus file not found: {corpus_path}")
        return False
    
    records = []
    errors = 0
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 5:
                    print(f"  Line {i}: JSON parse error: {e}")
    
    print(f"Records: {len(records)}, Errors: {errors}")
    
    required = ["doc_id", "text", "ocr_quality", "ocr_quality_tier", "word_count"]
    missing = Counter()
    for r in records:
        for field in required:
            if field not in r or r[field] is None:
                missing[field] += 1
    
    if missing:
        print(f"\nMissing/null fields:")
        for field, count in missing.most_common():
            print(f"  {field}: {count} records")
    
    empty = sum(1 for r in records if not r.get("text", "").strip())
    print(f"Empty text documents: {empty}")
    
    if records:
        print_corpus_stats(records)
    
    return errors == 0 and empty == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_corpus(data_dir: str | Path, output_path: str | Path,
                 sample_size: int | None = None, seed: int = 42):
    """Build the full corpus from extracted NLS data.
    
    Args:
        data_dir: Root directory of extracted data
        output_path: Output JSONL file path
        sample_size: If set, sample this many documents (stratified by OCR tier)
        seed: Random seed for reproducible sampling
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Discovering documents...")
    doc_infos = discover_documents(data_dir)
    print(f"Found {len(doc_infos)} documents with TXT files")
    
    if not doc_infos:
        print("No documents found! Check the data directory structure.")
        print(f"Expected layout: {data_dir}/collection/date_range/title/id/id.txt")
        return
    
    records = []
    skipped = 0
    errors = 0
    
    for doc_info in tqdm(doc_infos, desc="Processing documents"):
        try:
            record = process_document(doc_info)
            if record:
                records.append(record)
            else:
                skipped += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error processing {doc_info['doc_id']}: {e}")
    
    print(f"\nProcessed: {len(records)} documents")
    print(f"Skipped (empty/too short): {skipped}")
    print(f"Errors: {errors}")
    
    if sample_size and len(records) > sample_size:
        records = stratified_sample(records, sample_size, seed)
        print(f"Sampled {len(records)} documents (stratified by OCR tier)")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Corpus written to: {output_path}")
    print_corpus_stats(records)


def main():
    parser = argparse.ArgumentParser(description="Build NLS document corpus")
    parser.add_argument("--data-dir", type=str, default="data/extracted",
                       help="Root directory of extracted NLS data")
    parser.add_argument("--output", type=str, default="data/corpus.jsonl",
                       help="Output JSONL file path")
    parser.add_argument("--sample", type=int, default=None,
                       help="Sample N documents (stratified by OCR tier)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")
    parser.add_argument("--validate", action="store_true",
                       help="Validate an existing corpus file")
    parser.add_argument("--corpus", type=str, default=None,
                       help="Corpus file to validate (with --validate)")
    args = parser.parse_args()
    
    if args.validate:
        corpus_path = args.corpus or args.output
        validate_corpus(corpus_path)
    else:
        build_corpus(args.data_dir, args.output, args.sample, args.seed)


if __name__ == "__main__":
    main()
