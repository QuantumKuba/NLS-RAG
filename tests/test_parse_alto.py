"""Tests for ALTO XML parser."""

import json
import tempfile
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.parse_alto import parse_alto_file, parse_alto_directory, ALTOPage, ALTODocument


# Sample ALTO XML (NLS-like format)
SAMPLE_ALTO_V2 = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Description>
    <MeasurementUnit>pixel</MeasurementUnit>
    <OCRProcessing ID="OCR1">
      <ocrProcessingStep>
        <processingDateTime>2019-01-01</processingDateTime>
        <processingSoftware>
          <softwareName>ABBYY FineReader</softwareName>
        </processingSoftware>
      </ocrProcessingStep>
    </OCRProcessing>
  </Description>
  <Layout>
    <Page ID="P1" WIDTH="2400" HEIGHT="3200">
      <PrintSpace>
        <TextBlock ID="TB1">
          <TextLine ID="TL1">
            <String ID="S1" CONTENT="The" WC="0.95" CC="000" HPOS="100" VPOS="200" WIDTH="80" HEIGHT="30"/>
            <SP WIDTH="10"/>
            <String ID="S2" CONTENT="Union" WC="0.88" CC="0010" HPOS="190" VPOS="200" WIDTH="120" HEIGHT="30"/>
            <SP WIDTH="10"/>
            <String ID="S3" CONTENT="of" WC="0.99" CC="00" HPOS="320" VPOS="200" WIDTH="50" HEIGHT="30"/>
            <SP WIDTH="10"/>
            <String ID="S4" CONTENT="Crowns" WC="0.75" CC="002300" HPOS="380" VPOS="200" WIDTH="150" HEIGHT="30"/>
          </TextLine>
          <TextLine ID="TL2">
            <String ID="S5" CONTENT="was" WC="0.92" CC="000" HPOS="100" VPOS="250" WIDTH="80" HEIGHT="30"/>
            <SP WIDTH="10"/>
            <String ID="S6" CONTENT="signed" WC="0.85" CC="000100" HPOS="190" VPOS="250" WIDTH="140" HEIGHT="30"/>
            <SP WIDTH="10"/>
            <String ID="S7" CONTENT="in" WC="0.98" CC="00" HPOS="340" VPOS="250" WIDTH="40" HEIGHT="30"/>
            <SP WIDTH="10"/>
            <String ID="S8" CONTENT="1603" WC="0.65" CC="0090" HPOS="390" VPOS="250" WIDTH="100" HEIGHT="30"/>
          </TextLine>
        </TextBlock>
        <TextBlock ID="TB2">
          <TextLine ID="TL3">
            <String ID="S9" CONTENT="Scot1and" WC="0.45" CC="00090000" HPOS="100" VPOS="400" WIDTH="180" HEIGHT="30"/>
            <SP WIDTH="10"/>
            <String ID="S10" CONTENT="and" WC="0.98" CC="000" HPOS="290" VPOS="400" WIDTH="70" HEIGHT="30"/>
            <SP WIDTH="10"/>
            <String ID="S11" CONTENT="Eng1and" WC="0.42" CC="0009000" HPOS="370" VPOS="400" WIDTH="160" HEIGHT="30"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>"""


SAMPLE_ALTO_NO_NAMESPACE = """<?xml version="1.0" encoding="UTF-8"?>
<alto>
  <Layout>
    <Page ID="P1" WIDTH="1200" HEIGHT="1600">
      <PrintSpace>
        <TextBlock>
          <TextLine>
            <String CONTENT="Hello" WC="0.99" CC="00000"/>
            <SP/>
            <String CONTENT="World" WC="0.90" CC="00000"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>"""


class TestParseAltoFile:
    """Tests for single ALTO file parsing."""
    
    def test_basic_parsing(self, tmp_path):
        """Test basic ALTO XML parsing with v2 namespace."""
        alto_file = tmp_path / "page1.xml"
        alto_file.write_text(SAMPLE_ALTO_V2)
        
        page = parse_alto_file(alto_file)
        
        assert isinstance(page, ALTOPage)
        assert page.word_count > 0
        assert "The" in page.text
        assert "Union" in page.text
    
    def test_word_confidence_extraction(self, tmp_path):
        """Test that word confidence scores (WC) are correctly extracted."""
        alto_file = tmp_path / "page1.xml"
        alto_file.write_text(SAMPLE_ALTO_V2)
        
        page = parse_alto_file(alto_file)
        
        # Check that confidences exist
        confidences = [w.word_confidence for w in page.words if w.word_confidence is not None]
        assert len(confidences) > 0
        
        # All confidences should be in [0, 1]
        for c in confidences:
            assert 0.0 <= c <= 1.0
        
        # Check specific known values
        the_word = [w for w in page.words if w.content == "The"][0]
        assert the_word.word_confidence == 0.95
        
        scotland_word = [w for w in page.words if w.content == "Scot1and"][0]
        assert scotland_word.word_confidence == 0.45  # Low confidence = OCR error!
    
    def test_ocr_error_detection(self, tmp_path):
        """Test that OCR errors are preserved in text (important for evaluation)."""
        alto_file = tmp_path / "page1.xml"
        alto_file.write_text(SAMPLE_ALTO_V2)
        
        page = parse_alto_file(alto_file)
        
        # OCR errors should be preserved, not corrected
        assert "Scot1and" in page.text  # '1' instead of 'l'
        assert "Eng1and" in page.text   # '1' instead of 'l'
    
    def test_mean_confidence(self, tmp_path):
        """Test document-level confidence calculation."""
        alto_file = tmp_path / "page1.xml"
        alto_file.write_text(SAMPLE_ALTO_V2)
        
        page = parse_alto_file(alto_file)
        
        assert page.mean_word_confidence is not None
        assert 0.0 <= page.mean_word_confidence <= 1.0
    
    def test_page_dimensions(self, tmp_path):
        """Test page dimension extraction."""
        alto_file = tmp_path / "page1.xml"
        alto_file.write_text(SAMPLE_ALTO_V2)
        
        page = parse_alto_file(alto_file)
        
        assert page.width == 2400
        assert page.height == 3200
    
    def test_no_namespace(self, tmp_path):
        """Test parsing ALTO without namespace."""
        alto_file = tmp_path / "page1.xml"
        alto_file.write_text(SAMPLE_ALTO_NO_NAMESPACE)
        
        page = parse_alto_file(alto_file)
        
        assert "Hello" in page.text
        assert "World" in page.text
    
    def test_malformed_xml(self, tmp_path):
        """Test graceful handling of malformed XML."""
        alto_file = tmp_path / "broken.xml"
        alto_file.write_text("<alto><broken>not closed")
        
        page = parse_alto_file(alto_file)
        # Should not crash, return empty or partial
        assert isinstance(page, ALTOPage)
    
    def test_char_confidence(self, tmp_path):
        """Test character confidence (CC) extraction."""
        alto_file = tmp_path / "page1.xml"
        alto_file.write_text(SAMPLE_ALTO_V2)
        
        page = parse_alto_file(alto_file)
        
        # Check CC attribute preserved
        cc_words = [w for w in page.words if w.char_confidence is not None]
        assert len(cc_words) > 0


class TestParseAltoDirectory:
    """Tests for multi-page document parsing."""
    
    def test_multi_page(self, tmp_path):
        """Test combining multiple ALTO pages into one document."""
        # Create two page files
        (tmp_path / "page001.xml").write_text(SAMPLE_ALTO_V2)
        (tmp_path / "page002.xml").write_text(SAMPLE_ALTO_NO_NAMESPACE)
        
        doc = parse_alto_directory(tmp_path)
        
        assert isinstance(doc, ALTODocument)
        assert doc.page_count == 2
        assert doc.word_count > 0
        assert "Scot1and" in doc.text  # From page 1
        assert "Hello" in doc.text      # From page 2
    
    def test_ocr_quality_tier(self, tmp_path):
        """Test OCR quality tier assignment."""
        (tmp_path / "page001.xml").write_text(SAMPLE_ALTO_V2)
        
        doc = parse_alto_directory(tmp_path)
        
        assert doc.ocr_quality_tier in ["high", "medium", "low", "unknown"]
    
    def test_confidence_distribution(self, tmp_path):
        """Test confidence distribution calculation."""
        (tmp_path / "page001.xml").write_text(SAMPLE_ALTO_V2)
        
        doc = parse_alto_directory(tmp_path)
        dist = doc.confidence_distribution(bins=10)
        
        assert isinstance(dist, dict)
        assert len(dist) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
