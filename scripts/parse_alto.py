"""
ALTO XML Parser for NLS digitized collections.

Extracts OCR text and word-level confidence scores from ALTO XML files.
ALTO (Analysed Layout and Text Object) is used by NLS for storing OCR output
with layout information and confidence scores.

Usage:
    from scripts.parse_alto import parse_alto_file, parse_alto_directory
    
    # Single file
    result = parse_alto_file("path/to/alto.xml")
    
    # Directory of ALTO files (one document = multiple pages)
    doc = parse_alto_directory("path/to/document/alto_dir/")
"""

from __future__ import annotations

import os
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from lxml import etree


# Common ALTO namespaces across versions
ALTO_NAMESPACES = [
    "http://www.loc.gov/standards/alto/ns-v2#",
    "http://www.loc.gov/standards/alto/ns-v3#",
    "http://www.loc.gov/standards/alto/ns-v4#",
    "http://schema.ccs-gmbh.com/ALTO",
    "",  # no namespace fallback
]


@dataclass
class ALTOWord:
    """A single word extracted from ALTO XML."""
    content: str
    word_confidence: Optional[float] = None  # WC attribute (0-1, 1=confident)
    char_confidence: Optional[str] = None    # CC attribute (per-char, 0=confident, 9=unsure)


@dataclass
class ALTOPage:
    """A single page extracted from ALTO XML."""
    page_id: str = ""
    width: int = 0
    height: int = 0
    words: list = field(default_factory=list)
    
    @property
    def text(self) -> str:
        """Full text of the page, joining words with spaces."""
        return " ".join(w.content for w in self.words if w.content.strip())
    
    @property
    def word_count(self) -> int:
        return len([w for w in self.words if w.content.strip()])
    
    @property
    def mean_word_confidence(self) -> Optional[float]:
        """Mean word confidence for the page (0-1 scale, 1=confident)."""
        confidences = [w.word_confidence for w in self.words 
                      if w.word_confidence is not None]
        if confidences:
            return statistics.mean(confidences)
        return None
    
    @property
    def median_word_confidence(self) -> Optional[float]:
        confidences = [w.word_confidence for w in self.words 
                      if w.word_confidence is not None]
        if confidences:
            return statistics.median(confidences)
        return None


@dataclass
class ALTODocument:
    """A document composed of multiple ALTO pages."""
    doc_id: str = ""
    pages: list = field(default_factory=list)
    parse_errors: list = field(default_factory=list)
    
    @property
    def text(self) -> str:
        """Full text of all pages, separated by newlines."""
        return "\n\n".join(p.text for p in self.pages if p.text.strip())
    
    @property
    def word_count(self) -> int:
        return sum(p.word_count for p in self.pages)
    
    @property
    def page_count(self) -> int:
        return len(self.pages)
    
    @property
    def mean_word_confidence(self) -> Optional[float]:
        """Document-level mean word confidence."""
        all_confidences = []
        for page in self.pages:
            for w in page.words:
                if w.word_confidence is not None:
                    all_confidences.append(w.word_confidence)
        if all_confidences:
            return statistics.mean(all_confidences)
        return None
    
    @property
    def ocr_quality_tier(self) -> str:
        """Classify document OCR quality into tiers."""
        conf = self.mean_word_confidence
        if conf is None:
            return "unknown"
        if conf >= 0.90:
            return "high"
        elif conf >= 0.70:
            return "medium"
        else:
            return "low"
    
    @property
    def char_accuracy_estimate(self) -> Optional[float]:
        """Estimate character-level accuracy from word confidences.
        
        This is approximate: word confidence correlates with but doesn't
        directly equal character accuracy. We use it as a proxy.
        """
        return self.mean_word_confidence  # Simplified proxy
    
    def confidence_distribution(self, bins: int = 10) -> dict:
        """Get distribution of word confidences across bins."""
        all_confidences = []
        for page in self.pages:
            for w in page.words:
                if w.word_confidence is not None:
                    all_confidences.append(w.word_confidence)
        
        if not all_confidences:
            return {}
        
        bin_size = 1.0 / bins
        distribution = {}
        for i in range(bins):
            lower = i * bin_size
            upper = (i + 1) * bin_size
            label = f"{lower:.1f}-{upper:.1f}"
            count = sum(1 for c in all_confidences if lower <= c < upper)
            distribution[label] = count
        
        # Include 1.0 in the last bin
        last_label = f"{(bins-1)*bin_size:.1f}-1.0"
        distribution[last_label] += sum(1 for c in all_confidences if c == 1.0)
        
        return distribution


def _find_namespace(root) -> str:
    """Detect the ALTO namespace used in the XML file."""
    tag = root.tag
    if "}" in tag:
        ns = tag.split("}")[0].strip("{")
        return ns
    
    # Check nsmap
    for prefix, uri in root.nsmap.items():
        if "alto" in uri.lower():
            return uri
    
    return ""


def _ns(namespace: str, tag: str) -> str:
    """Create a namespaced tag string."""
    if namespace:
        return f"{{{namespace}}}{tag}"
    return tag


def parse_alto_file(filepath: str | Path) -> ALTOPage:
    """Parse a single ALTO XML file and extract text with confidence scores.
    
    Args:
        filepath: Path to an ALTO XML file
        
    Returns:
        ALTOPage with extracted words and confidence scores
    """
    filepath = Path(filepath)
    page = ALTOPage(page_id=filepath.stem)
    
    try:
        tree = etree.parse(str(filepath))
        root = tree.getroot()
    except etree.XMLSyntaxError as e:
        # Try to recover from malformed XML
        try:
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(str(filepath), parser)
            root = tree.getroot()
        except Exception:
            page.words = []
            return page
    
    namespace = _find_namespace(root)
    
    # Find all String elements (words)
    string_tag = _ns(namespace, "String")
    
    # Try XPath with namespace
    if namespace:
        nsmap = {"alto": namespace}
        strings = root.xpath("//alto:String", namespaces=nsmap)
    else:
        strings = root.xpath("//String")
    
    # If XPath fails, try iter
    if not strings:
        strings = list(root.iter(string_tag))
    
    # Also try without namespace as fallback
    if not strings:
        strings = list(root.iter("String"))
    
    for s in strings:
        content = s.get("CONTENT", "")
        
        # Parse word confidence (WC): float 0-1
        wc_str = s.get("WC")
        wc = None
        if wc_str:
            try:
                wc = float(wc_str)
            except (ValueError, TypeError):
                pass
        
        # Parse character confidence (CC): string of digits
        cc = s.get("CC")
        
        word = ALTOWord(content=content, word_confidence=wc, char_confidence=cc)
        page.words.append(word)
    
    # Try to get page dimensions
    page_elem = root.find(f".//{_ns(namespace, 'Page')}")
    if page_elem is None:
        page_elem = root.find(".//Page")
    if page_elem is not None:
        try:
            page.width = int(page_elem.get("WIDTH", 0))
            page.height = int(page_elem.get("HEIGHT", 0))
        except (ValueError, TypeError):
            pass
    
    return page


def parse_alto_directory(dir_path: str | Path, sort_pages: bool = True) -> ALTODocument:
    """Parse all ALTO XML files in a directory as a single document.
    
    NLS typically stores one ALTO XML per page. This function combines
    all pages into a single document.
    
    Args:
        dir_path: Directory containing ALTO XML files
        sort_pages: Sort pages by filename (assumes numeric naming)
        
    Returns:
        ALTODocument with all pages combined
    """
    dir_path = Path(dir_path)
    doc = ALTODocument(doc_id=dir_path.name)
    
    # Find all XML files
    xml_files = sorted(dir_path.glob("*.xml")) if sort_pages else list(dir_path.glob("*.xml"))
    
    # Also check for files without extension that might be ALTO
    if not xml_files:
        xml_files = sorted(dir_path.glob("*")) if sort_pages else list(dir_path.glob("*"))
        xml_files = [f for f in xml_files if f.is_file() and not f.name.startswith(".")]
    
    for xml_file in xml_files:
        try:
            page = parse_alto_file(xml_file)
            doc.pages.append(page)
        except Exception as e:
            doc.parse_errors.append(f"{xml_file.name}: {str(e)}")
    
    return doc


def extract_text_and_quality(dir_path: str | Path) -> dict:
    """Convenience function: extract text and OCR quality from a document directory.
    
    Returns a dict ready for corpus building:
    {
        "text": str,
        "ocr_quality": float,
        "ocr_quality_tier": str,
        "word_count": int,
        "page_count": int,
        "confidence_distribution": dict,
    }
    """
    doc = parse_alto_directory(dir_path)
    
    return {
        "text": doc.text,
        "ocr_quality": doc.mean_word_confidence,
        "ocr_quality_tier": doc.ocr_quality_tier,
        "word_count": doc.word_count,
        "page_count": doc.page_count,
        "confidence_distribution": doc.confidence_distribution(),
        "parse_errors": doc.parse_errors,
    }


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Parse ALTO XML files")
    parser.add_argument("path", help="Path to ALTO XML file or directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file():
        page = parse_alto_file(path)
        if args.json:
            print(json.dumps({
                "page_id": page.page_id,
                "word_count": page.word_count,
                "mean_confidence": page.mean_word_confidence,
                "text_preview": page.text[:500],
            }, indent=2))
        else:
            print(f"Page: {page.page_id}")
            print(f"Words: {page.word_count}")
            print(f"Mean confidence: {page.mean_word_confidence}")
            print(f"Text preview: {page.text[:300]}...")
    
    elif path.is_dir():
        result = extract_text_and_quality(path)
        if args.json:
            # Don't dump full text to stdout
            result_summary = {k: v for k, v in result.items() if k != "text"}
            result_summary["text_length"] = len(result["text"])
            result_summary["text_preview"] = result["text"][:500]
            print(json.dumps(result_summary, indent=2))
        else:
            print(f"Document: {path.name}")
            print(f"Pages: {result['page_count']}")
            print(f"Words: {result['word_count']}")
            print(f"OCR Quality: {result['ocr_quality']}")
            print(f"Tier: {result['ocr_quality_tier']}")
            print(f"Text preview: {result['text'][:300]}...")
