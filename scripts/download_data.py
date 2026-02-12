"""
Download NLS-CH-Multimodal dataset from HuggingFace and extract archives.

Downloads RAR archives from NeuraSearchLab/NLS-CH-Multimodal and extracts
them into the local data directory for processing.

Usage:
    python scripts/download_data.py [--collection COLLECTION] [--output-dir DIR]
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


REPO_ID = "NeuraSearchLab/NLS-CH-Multimodal"
REPO_TYPE = "dataset"

# Known collections and their archive files
COLLECTIONS = {
    "africa_and_new_imperialism": [
        "africa_and_new_imperialism/Africa_And_New_Imperialism_1_2/Africa_And_New_Imperialism 1.rar"
    ],
    "indiaraj": [
        "indiaraj/indiaraj 1.rar",
        "indiaraj/indiaraj 2.rar",
    ],
}


def list_available_collections():
    """List all available collections from the HuggingFace repo."""
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    
    # Group files by top-level directory
    collections = {}
    for f in files:
        if f == ".gitattributes":
            continue
        parts = f.split("/")
        if len(parts) >= 1:
            col = parts[0]
            if col not in collections:
                collections[col] = []
            collections[col].append(f)
    
    return collections


def download_collection(collection_name: str, output_dir: Path):
    """Download a single collection's archives from HuggingFace."""
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    if collection_name in COLLECTIONS:
        files = COLLECTIONS[collection_name]
    else:
        # Dynamically discover files for unknown collections
        all_collections = list_available_collections()
        if collection_name not in all_collections:
            print(f"Error: Collection '{collection_name}' not found.")
            print(f"Available: {list(all_collections.keys())}")
            sys.exit(1)
        files = [f for f in all_collections[collection_name] if f.endswith(".rar")]
    
    print(f"\n{'='*60}")
    print(f"Downloading collection: {collection_name}")
    print(f"Files to download: {len(files)}")
    print(f"{'='*60}\n")
    
    downloaded_files = []
    for filepath in files:
        print(f"Downloading: {filepath}")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filepath,
            local_dir=str(raw_dir),
        )
        downloaded_files.append(Path(local_path))
        print(f"  → Saved to: {local_path}")
    
    return downloaded_files


def extract_rar(rar_path: Path, extract_dir: Path):
    """Extract a RAR archive using the system's unrar command or Python rarfile."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExtracting: {rar_path.name}")
    print(f"  → To: {extract_dir}")
    
    # Try system unrar first (faster, handles large files better)
    try:
        result = subprocess.run(
            ["unrar", "x", "-o+", str(rar_path), str(extract_dir) + "/"],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for large archives
        )
        if result.returncode == 0:
            print(f"  ✓ Extracted successfully with unrar")
            return True
        else:
            print(f"  ✗ unrar failed: {result.stderr[:200]}")
    except FileNotFoundError:
        print("  ! unrar not found, trying Python rarfile...")
    except subprocess.TimeoutExpired:
        print("  ! Extraction timed out after 1 hour")
        return False
    
    # Fallback to Python rarfile
    try:
        import rarfile
        with rarfile.RarFile(str(rar_path)) as rf:
            rf.extractall(str(extract_dir))
        print(f"  ✓ Extracted successfully with rarfile")
        return True
    except Exception as e:
        print(f"  ✗ Failed to extract: {e}")
        print("  ! Install unrar: brew install unrar (macOS) or apt install unrar (Linux)")
        return False


def scan_extracted_data(extract_dir: Path):
    """Scan extracted data to understand structure and report statistics."""
    stats = {
        "alto_files": 0,
        "mets_files": 0,
        "text_files": 0,
        "image_files": 0,
        "other_files": 0,
        "total_size_mb": 0,
    }
    
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            filepath = Path(root) / f
            size_mb = filepath.stat().st_size / (1024 * 1024)
            stats["total_size_mb"] += size_mb
            
            f_lower = f.lower()
            if f_lower.endswith(".xml"):
                # Distinguish ALTO from METS by content heuristic
                if "alto" in f_lower or f_lower.startswith("0"):
                    stats["alto_files"] += 1
                elif "mets" in f_lower:
                    stats["mets_files"] += 1
                else:
                    # Quick peek to classify
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                            header = fh.read(500)
                            if "alto" in header.lower():
                                stats["alto_files"] += 1
                            elif "mets" in header.lower():
                                stats["mets_files"] += 1
                            else:
                                stats["other_files"] += 1
                    except Exception:
                        stats["other_files"] += 1
            elif f_lower.endswith(".txt"):
                stats["text_files"] += 1
            elif f_lower.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                stats["image_files"] += 1
            else:
                stats["other_files"] += 1
    
    print(f"\n{'='*60}")
    print(f"Extracted Data Summary:")
    print(f"{'='*60}")
    print(f"  ALTO XML files:  {stats['alto_files']:,}")
    print(f"  METS XML files:  {stats['mets_files']:,}")
    print(f"  Text files:      {stats['text_files']:,}")
    print(f"  Image files:     {stats['image_files']:,}")
    print(f"  Other files:     {stats['other_files']:,}")
    print(f"  Total size:      {stats['total_size_mb']:,.1f} MB")
    print(f"{'='*60}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download NLS-CH-Multimodal dataset")
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection to download (e.g., 'indiaraj', 'africa_and_new_imperialism'). "
             "If not specified, lists available collections.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for downloaded data (default: data/)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, only extract existing archives",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction, only download archives",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan already-extracted data and report statistics",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.scan_only:
        extract_dir = output_dir / "extracted"
        if extract_dir.exists():
            scan_extracted_data(extract_dir)
        else:
            print(f"No extracted data found at {extract_dir}")
        return
    
    if args.collection is None:
        print("Available collections:")
        for name, files in COLLECTIONS.items():
            print(f"  {name}: {len(files)} archive(s)")
        print("\nUse --collection NAME to download a specific collection.")
        
        # Also check HuggingFace for any new collections
        print("\nChecking HuggingFace for updates...")
        try:
            remote = list_available_collections()
            for name in remote:
                if name not in COLLECTIONS:
                    print(f"  [NEW] {name}: {len(remote[name])} file(s)")
        except Exception as e:
            print(f"  (Could not connect to HuggingFace: {e})")
        return
    
    # Download
    if not args.skip_download:
        downloaded = download_collection(args.collection, output_dir)
    else:
        # Find existing archives
        raw_dir = output_dir / "raw"
        downloaded = list(raw_dir.rglob("*.rar"))
        print(f"Found {len(downloaded)} existing archive(s)")
    
    # Extract
    if not args.skip_extract:
        extract_dir = output_dir / "extracted" / args.collection
        for rar_path in downloaded:
            extract_rar(rar_path, extract_dir)
        
        # Scan extracted data
        scan_extracted_data(extract_dir)
    
    print("Done!")


if __name__ == "__main__":
    main()
