"""
Resolve git merge conflicts in verification_template.json.

Parses the conflicted file, extracts both HEAD and main versions of each
query, and merges verified_by lists. Produces a clean JSON file.

Usage:
    python scripts/resolve_conflicts.py
    python scripts/resolve_conflicts.py --input data/queries/verification_template.json
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path


def strip_conflicts_and_extract(raw: str) -> str:
    """Remove git conflict markers, keeping both sides' content.
    
    Strategy: for each conflict block, keep the HEAD version (your edits)
    but merge verified_by from both sides.
    """
    # Remove all conflict markers, keeping content from both sides
    # Pattern: <<<<<<< HEAD ... ======= ... >>>>>>> main
    
    # First pass: remove just the marker lines
    lines = raw.split('\n')
    clean_lines = []
    in_head = False
    in_main = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('<<<<<<< '):
            in_head = True
            continue
        elif stripped == '=======' and (in_head or in_main):
            in_head = False
            in_main = True
            continue
        elif stripped.startswith('>>>>>>> '):
            in_main = False
            continue
        
        if in_main:
            # Skip the 'main' side — we'll merge verified_by separately
            continue
        
        clean_lines.append(line)
    
    return '\n'.join(clean_lines)


def extract_all_queries_from_raw(raw: str) -> dict[str, list[str]]:
    """Extract all verified_by entries keyed by doc_id from raw text (both sides)."""
    reviewers_by_doc = {}
    
    # Find all doc_id + verified_by pairs in the raw text
    # Look for patterns like "doc_id": "XXXX" followed by verified_by lists
    blocks = re.split(r'\{', raw)
    
    current_doc_id = None
    for block in blocks:
        doc_match = re.search(r'"doc_id"\s*:\s*"(\d+)"', block)
        if doc_match:
            current_doc_id = doc_match.group(1)
        
        vb_match = re.search(r'"verified_by"\s*:\s*\[(.*?)\]', block, re.DOTALL)
        if vb_match and current_doc_id:
            names_raw = vb_match.group(1)
            names = re.findall(r'"([^"]+)"', names_raw)
            if current_doc_id not in reviewers_by_doc:
                reviewers_by_doc[current_doc_id] = set()
            reviewers_by_doc[current_doc_id].update(names)
    
    return {k: sorted(v) for k, v in reviewers_by_doc.items()}


def main():
    parser = argparse.ArgumentParser(description="Resolve merge conflicts in verification template")
    parser.add_argument("--input", type=str,
                        default="data/queries/verification_template.json",
                        help="Path to conflicted verification template")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: overwrite input)")
    args = parser.parse_args()
    
    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    
    raw = path.read_text(encoding="utf-8")
    
    # Check if there are actually conflicts
    if '=======' not in raw:
        print("No merge conflicts detected.")
        sys.exit(0)
    
    # Count conflicts
    n_conflicts = raw.count('=======')
    print(f"Found {n_conflicts} merge conflict markers")
    
    # Step 1: Extract all reviewer info from both sides before cleaning
    all_reviewers = extract_all_queries_from_raw(raw)
    print(f"Extracted reviewer info for {len(all_reviewers)} documents")
    for doc_id, reviewers in list(all_reviewers.items())[:5]:
        print(f"  doc {doc_id}: {reviewers}")
    if len(all_reviewers) > 5:
        print(f"  ... and {len(all_reviewers) - 5} more")
    
    # Step 2: Strip conflict markers (keep HEAD side)
    clean = strip_conflicts_and_extract(raw)
    
    # Step 3: Try to parse the cleaned JSON
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"\nError: cleaned JSON is still invalid: {e}")
        # Save the intermediate for debugging
        debug_path = path.with_suffix('.cleaned.txt')
        debug_path.write_text(clean, encoding="utf-8")
        print(f"Saved intermediate to {debug_path} for inspection")
        sys.exit(1)
    
    # Step 4: Merge verified_by from both sides
    merged_count = 0
    for section_key in ["retrieval_queries", "rag_questions"]:
        for item in data.get(section_key, []):
            doc_id = item.get("doc_id")
            if doc_id in all_reviewers:
                existing = set(item.get("verified_by", []))
                combined = existing | set(all_reviewers[doc_id])
                if combined != existing:
                    merged_count += 1
                item["verified_by"] = sorted(combined)
                
                # Update status based on reviewer count
                if len(item["verified_by"]) >= 2:
                    if item.get("status") != "rejected":
                        item["status"] = "verified"
                elif len(item["verified_by"]) == 1:
                    if item.get("status") != "rejected":
                        item["status"] = "partially_verified"
    
    print(f"Merged reviewer annotations for {merged_count} items")
    
    # Step 5: Report
    for section_key in ["retrieval_queries", "rag_questions"]:
        items = data.get(section_key, [])
        statuses = {}
        for item in items:
            s = item.get("status", "unknown")
            statuses[s] = statuses.get(s, 0) + 1
        print(f"\n{section_key}: {len(items)} items")
        for status, count in sorted(statuses.items()):
            print(f"  {status}: {count}")
    
    # Step 6: Save
    out_path = Path(args.output) if args.output else path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved resolved file to: {out_path}")
    print(f"   You can now run: python scripts/compute_agreement.py")


if __name__ == "__main__":
    main()
