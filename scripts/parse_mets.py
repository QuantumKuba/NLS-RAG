"""
METS XML Parser for NLS digitized collections.

Extracts bibliographic metadata from METS (Metadata Encoding and Transmission
Standard) XML files used by the National Library of Scotland.

METS files contain:
- Descriptive metadata (dmdSec): title, author, date via MODS
- Administrative metadata (amdSec): rights, capture info
- File listing (fileSec): associated ALTO, image, text files
- Structural metadata (structMap): document hierarchy

Usage:
    from scripts.parse_mets import parse_mets_file
    metadata = parse_mets_file("path/to/mets.xml")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from lxml import etree


# Namespace map for METS and associated schemas
NS = {
    "mets": "http://www.loc.gov/METS/",
    "mods": "http://www.loc.gov/mods/v3",
    "premis": "info:lc/xmlns/premis-v2",
    "xlink": "http://www.w3.org/1999/xlink",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
}


@dataclass
class METSMetadata:
    """Metadata extracted from a METS XML file."""
    doc_id: str = ""
    title: str = ""
    author: str = ""
    date: str = ""
    date_numeric: Optional[int] = None  # Extracted year as integer
    publisher: str = ""
    place_of_publication: str = ""
    language: str = ""
    subject: list = field(default_factory=list)
    genre: str = ""
    collection: str = ""
    physical_description: str = ""
    extent: str = ""
    
    # File references
    alto_files: list = field(default_factory=list)
    image_files: list = field(default_factory=list)
    text_files: list = field(default_factory=list)
    
    # Structure
    page_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "author": self.author,
            "date": self.date,
            "date_numeric": self.date_numeric,
            "publisher": self.publisher,
            "place_of_publication": self.place_of_publication,
            "language": self.language,
            "subject": self.subject,
            "genre": self.genre,
            "collection": self.collection,
            "physical_description": self.physical_description,
            "extent": self.extent,
            "alto_file_count": len(self.alto_files),
            "image_file_count": len(self.image_files),
            "page_count": self.page_count,
        }


def _extract_year(date_str: str) -> Optional[int]:
    """Extract a 4-digit year from various date formats."""
    if not date_str:
        return None
    
    # Match 4-digit year
    match = re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', date_str)
    if match:
        return int(match.group(1))
    return None


def _get_text(element, xpath: str, namespaces: dict = None) -> str:
    """Safely extract text from an XPath query."""
    if element is None:
        return ""
    
    ns = namespaces or NS
    results = element.xpath(xpath, namespaces=ns)
    
    if not results:
        return ""
    
    result = results[0]
    if isinstance(result, str):
        return result.strip()
    elif hasattr(result, "text") and result.text:
        return result.text.strip()
    elif hasattr(result, "itertext"):
        return " ".join(result.itertext()).strip()
    return ""


def _get_all_text(element, xpath: str, namespaces: dict = None) -> list:
    """Extract all text matches from an XPath query."""
    if element is None:
        return []
    
    ns = namespaces or NS
    results = element.xpath(xpath, namespaces=ns)
    
    texts = []
    for result in results:
        if isinstance(result, str):
            texts.append(result.strip())
        elif hasattr(result, "text") and result.text:
            texts.append(result.text.strip())
        elif hasattr(result, "itertext"):
            texts.append(" ".join(result.itertext()).strip())
    
    return [t for t in texts if t]


def parse_mets_file(filepath: str | Path) -> METSMetadata:
    """Parse a METS XML file and extract metadata.
    
    Args:
        filepath: Path to a METS XML file
        
    Returns:
        METSMetadata with extracted bibliographic and structural information
    """
    filepath = Path(filepath)
    metadata = METSMetadata(doc_id=filepath.stem)
    
    try:
        tree = etree.parse(str(filepath))
        root = tree.getroot()
    except etree.XMLSyntaxError:
        try:
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(str(filepath), parser)
            root = tree.getroot()
        except Exception:
            return metadata
    
    # Detect namespace - try with and without standard METS namespace
    has_mets_ns = bool(root.xpath("//mets:dmdSec", namespaces=NS))
    
    # ─── Extract from MODS descriptive metadata ───
    if has_mets_ns:
        mods_root = root.xpath("//mets:dmdSec//mods:mods", namespaces=NS)
    else:
        mods_root = root.xpath("//mods:mods", namespaces=NS)
    
    if not mods_root:
        # Try without namespaces
        mods_root = root.xpath("//*[local-name()='mods']")
    
    mods = mods_root[0] if mods_root else None
    
    if mods is not None:
        # Title
        metadata.title = _get_text(mods, ".//mods:titleInfo/mods:title")
        if not metadata.title:
            metadata.title = _get_text(mods, ".//*[local-name()='title']")
        
        # Author
        metadata.author = _get_text(mods, ".//mods:name/mods:namePart")
        if not metadata.author:
            metadata.author = _get_text(mods, ".//*[local-name()='namePart']")
        
        # Date
        metadata.date = _get_text(mods, ".//mods:originInfo/mods:dateIssued")
        if not metadata.date:
            metadata.date = _get_text(mods, ".//mods:originInfo/mods:dateCreated")
        if not metadata.date:
            metadata.date = _get_text(mods, ".//*[local-name()='dateIssued']")
        metadata.date_numeric = _extract_year(metadata.date)
        
        # Publisher
        metadata.publisher = _get_text(mods, ".//mods:originInfo/mods:publisher")
        if not metadata.publisher:
            metadata.publisher = _get_text(mods, ".//*[local-name()='publisher']")
        
        # Place of publication
        metadata.place_of_publication = _get_text(
            mods, ".//mods:originInfo/mods:place/mods:placeTerm"
        )
        
        # Language
        metadata.language = _get_text(mods, ".//mods:language/mods:languageTerm")
        
        # Subjects
        metadata.subject = _get_all_text(mods, ".//mods:subject/mods:topic")
        if not metadata.subject:
            metadata.subject = _get_all_text(mods, ".//*[local-name()='topic']")
        
        # Genre
        metadata.genre = _get_text(mods, ".//mods:genre")
        
        # Physical description
        metadata.extent = _get_text(mods, ".//mods:physicalDescription/mods:extent")
    
    # ─── Extract file references from fileSec ───
    if has_mets_ns:
        file_groups = root.xpath("//mets:fileSec/mets:fileGrp", namespaces=NS)
    else:
        file_groups = root.xpath("//*[local-name()='fileGrp']")
    
    for fg in file_groups:
        use = fg.get("USE", "").lower()
        
        # Get file locations
        if has_mets_ns:
            files = fg.xpath(".//mets:file/mets:FLocat", namespaces=NS)
        else:
            files = fg.xpath(".//*[local-name()='FLocat']")
        
        for f in files:
            href = f.get(f"{{{NS['xlink']}}}href", "")
            if not href:
                href = f.get("href", "")
            
            if not href:
                continue
            
            if "alto" in use or "xml" in use or href.lower().endswith(".xml"):
                metadata.alto_files.append(href)
            elif any(ext in href.lower() for ext in [".jpg", ".jpeg", ".tif", ".tiff", ".png"]):
                metadata.image_files.append(href)
            elif href.lower().endswith(".txt"):
                metadata.text_files.append(href)
    
    # ─── Extract page count from structMap ───
    if has_mets_ns:
        divs = root.xpath("//mets:structMap//mets:div[@TYPE='page' or @TYPE='Page']",
                         namespaces=NS)
    else:
        divs = root.xpath("//*[local-name()='div'][@TYPE='page' or @TYPE='Page']")
    
    metadata.page_count = len(divs) if divs else len(metadata.alto_files)
    
    return metadata


def find_mets_file(doc_dir: str | Path) -> Optional[Path]:
    """Find the METS XML file in a document directory.
    
    NLS typically names the METS file with a specific pattern.
    """
    doc_dir = Path(doc_dir)
    
    # Common METS filename patterns
    patterns = [
        "*METS*.xml", "*mets*.xml",
        "METS.xml", "mets.xml",
        "*_mets.xml", "*_METS.xml",
    ]
    
    for pattern in patterns:
        matches = list(doc_dir.glob(pattern))
        if matches:
            return matches[0]
    
    # Fallback: look for XML files that contain METS in content
    for xml_file in doc_dir.glob("*.xml"):
        try:
            with open(xml_file, "r", encoding="utf-8", errors="ignore") as f:
                header = f.read(1000)
                if "METS" in header or "mets" in header:
                    return xml_file
        except Exception:
            continue
    
    return None


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Parse METS XML files")
    parser.add_argument("path", help="Path to METS XML file or directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file():
        meta = parse_mets_file(path)
    elif path.is_dir():
        mets_file = find_mets_file(path)
        if mets_file:
            meta = parse_mets_file(mets_file)
            print(f"Found METS file: {mets_file.name}")
        else:
            print("No METS file found in directory.")
            exit(1)
    else:
        print(f"Path not found: {path}")
        exit(1)
    
    if args.json:
        print(json.dumps(meta.to_dict(), indent=2))
    else:
        d = meta.to_dict()
        for k, v in d.items():
            print(f"  {k}: {v}")
