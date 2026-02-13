"""
Compute inter-annotator agreement on query verification.

Reads verification_template.json after both reviewers have completed their
pass, computes Cohen's kappa, and outputs a summary including which queries
to keep, reject, or adjudicate.

Usage:
    python scripts/compute_agreement.py                       # defaults
    python scripts/compute_agreement.py --input data/queries/verification_template.json
    python scripts/compute_agreement.py --export              # write approved sets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter


def _decision(item: dict, reviewer: str) -> str | None:
    """Infer a reviewer's decision from the item data.
    
    Detection priority:
    1. Explicit per-reviewer 'decisions' dict (new format) → definitive
    2. Explicit rejection note from this reviewer in reviewer_notes → reject
    3. Item status is 'rejected' and reviewer is in verified_by → infer reject
    4. Reviewer in verified_by with non-rejected status → accept
    5. Reviewer not in verified_by → None (not yet reviewed)
    """
    if reviewer not in item.get("verified_by", []):
        return None
    
    # Priority 1: explicit decisions dict (most reliable)
    decisions = item.get("decisions", {})
    if reviewer in decisions:
        return decisions[reviewer]
    
    # Priority 2: check reviewer_notes for rejection
    notes = item.get("reviewer_notes", [])
    for note in notes:
        if note.startswith(f"{reviewer}:"):
            return "reject"
    
    # Priority 3: use item status as fallback
    status = item.get("status", "")
    if status == "rejected":
        verified_by = item.get("verified_by", [])
        if len(verified_by) == 1:
            return "reject"
        # Multiple reviewers, check if another reviewer has a rejection note
        other_rejected = any(
            note.startswith(f"{other}:")
            for note in notes
            for other in verified_by
            if other != reviewer
        )
        if other_rejected:
            return "accept"  # Other reviewer rejected, this one didn't
        return "reject"  # Conservative: can't distinguish
    
    return "accept"


def cohens_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Compute Cohen's kappa without sklearn dependency."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 0.0

    # Observed agreement
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n

    # Expected agreement by chance
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    all_labels = set(labels_a) | set(labels_b)
    p_e = sum((count_a.get(k, 0) / n) * (count_b.get(k, 0) / n) for k in all_labels)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def analyse(template_path: Path, export: bool = False):
    with open(template_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sections = [
        ("Retrieval Queries", data.get("retrieval_queries", []), "query_id", "query"),
        ("RAG Questions", data.get("rag_questions", []), "question_id", "question"),
    ]

    all_approved_retrieval = []
    all_approved_rag = []

    for section_name, items, id_key, text_key in sections:
        print(f"\n{'='*60}")
        print(f"  {section_name} ({len(items)} items)")
        print(f"{'='*60}")

        # Collect all reviewers
        reviewers = set()
        for item in items:
            for r in item.get("verified_by", []):
                reviewers.add(r)

        if len(reviewers) < 2:
            print(f"\n  ⚠  Only {len(reviewers)} reviewer(s) found: {reviewers or '{none}'}")
            print(f"  Need 2 reviewers to compute agreement.\n")
            # Still report status breakdown
            statuses = Counter(item.get("status", "unknown") for item in items)
            for status, count in statuses.most_common():
                print(f"    {status}: {count}")
            continue

        reviewers = sorted(reviewers)
        r1, r2 = reviewers[0], reviewers[1]
        print(f"\n  Reviewers: {r1}, {r2}")

        # Classify each item
        both_accept = []
        both_reject = []
        disagree = []
        incomplete = []

        labels_r1 = []
        labels_r2 = []

        for item in items:
            d1 = _decision(item, r1)
            d2 = _decision(item, r2)

            if d1 is None or d2 is None:
                incomplete.append(item)
                continue

            labels_r1.append(1 if d1 == "accept" else 0)
            labels_r2.append(1 if d2 == "accept" else 0)

            if d1 == "accept" and d2 == "accept":
                both_accept.append(item)
            elif d1 == "reject" and d2 == "reject":
                both_reject.append(item)
            else:
                disagree.append(item)

        # Results
        n_rated = len(labels_r1)
        print(f"\n  Items rated by both: {n_rated}")
        print(f"  Incomplete (missing a review): {len(incomplete)}")

        if n_rated > 0:
            kappa = cohens_kappa(labels_r1, labels_r2)
            raw_agreement = sum(1 for a, b in zip(labels_r1, labels_r2) if a == b) / n_rated

            print(f"\n  ┌─────────────────────────────┐")
            print(f"  │ Raw agreement:  {raw_agreement:.1%}        │")
            print(f"  │ Cohen's κ:      {kappa:.3f}        │")
            print(f"  └─────────────────────────────┘")

            if kappa > 0.8:
                interpretation = "Almost perfect agreement"
            elif kappa > 0.6:
                interpretation = "Substantial agreement"
            elif kappa > 0.4:
                interpretation = "Moderate agreement"
            elif kappa > 0.2:
                interpretation = "Fair agreement"
            else:
                interpretation = "Slight agreement"
            print(f"  Interpretation: {interpretation}")

        print(f"\n  ✅ Both accept:  {len(both_accept)}")
        print(f"  ❌ Both reject:  {len(both_reject)}")
        print(f"  ⚠️  Disagree:    {len(disagree)}")

        if both_reject:
            print(f"\n  Rejected items:")
            for item in both_reject:
                print(f"    - [{item[id_key]}] {item[text_key][:80]}...")

        if disagree:
            print(f"\n  Items needing adjudication:")
            for item in disagree:
                d1 = _decision(item, r1)
                d2 = _decision(item, r2)
                print(f"    - [{item[id_key]}] {r1}={d1}, {r2}={d2}")
                print(f"      {item[text_key][:80]}...")
                for note in item.get("reviewer_notes", []):
                    print(f"      Note: {note}")

        if incomplete:
            print(f"\n  Incomplete items (need review):")
            for item in incomplete:
                print(f"    - [{item[id_key]}] verified_by={item.get('verified_by', [])}")

        # Collect approved items
        approved = both_accept  # conservative: only both-accept
        if section_name == "Retrieval Queries":
            all_approved_retrieval = approved
        else:
            all_approved_rag = approved

    # Export approved sets
    if export:
        out_dir = template_path.parent
        if all_approved_retrieval:
            with open(out_dir / "queries_verified.json", "w", encoding="utf-8") as f:
                json.dump(all_approved_retrieval, f, indent=2, ensure_ascii=False)
            # Updated qrels
            with open(out_dir / "qrels_verified.tsv", "w") as f:
                f.write("query-id\tcorpus-id\tscore\n")
                for q in all_approved_retrieval:
                    f.write(f"{q['query_id']}\t{q['doc_id']}\t1\n")
            print(f"\n  Exported {len(all_approved_retrieval)} approved retrieval queries")

        if all_approved_rag:
            with open(out_dir / "rag_questions_verified.json", "w", encoding="utf-8") as f:
                json.dump(all_approved_rag, f, indent=2, ensure_ascii=False)
            print(f"  Exported {len(all_approved_rag)} approved RAG questions")

    # Summary for paper
    print(f"\n{'='*60}")
    print(f"  PAPER-READY SUMMARY")
    print(f"{'='*60}")
    print(f"  Use this in your LaTeX:")
    print(f"  \"Inter-annotator agreement was [κ value] (Cohen's κ),")
    print(f"   with N items resolved through adjudication.\"")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Compute verification agreement")
    parser.add_argument("--input", type=str,
                        default="data/queries/verification_template.json",
                        help="Path to verification template")
    parser.add_argument("--export", action="store_true",
                        help="Export approved queries to separate files")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    analyse(path, args.export)


if __name__ == "__main__":
    main()
