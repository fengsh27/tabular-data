from TabFuncFlow.utils.table_utils import dataframe_to_markdown
from extractor.pmid_extractor.html_table_extractor import HtmlTableExtractor


def test_HtmlTableExtractor_31206433():
    extractor = HtmlTableExtractor()
    with open("./tests/data/31206433.html", "r") as fobj:
        html = fobj.read()
        tables = extractor.extract_tables(html)
        assert len(tables) == 5
        assert len(tables[0]["caption"]) == 0
        assert len(tables[0]["footnote"]) == 0
        assert len(tables[1]["caption"]) > 0
        assert len(tables[1]["footnote"]) > 0
        assert len(tables[2]["caption"]) > 0
        assert len(tables[2]["footnote"]) > 0
        assert len(tables[3]["caption"]) > 0
        assert len(tables[3]["footnote"]) > 0
        assert len(tables[4]["caption"]) == 0
        assert len(tables[4]["footnote"]) == 0

        title = extractor.extract_title(html)
        assert title is not None


def test_HtmlTableExtractor_17158945():
    extractor = HtmlTableExtractor()
    with open("./tests/data/17158945.html", "r") as fobj:
        html = fobj.read()
        tables = extractor.extract_tables(html)
        assert len(tables) > 0

        title = extractor.extract_title(html)
        assert title is not None


def test_HtmlTableExtractor_15601346():
    extractor = HtmlTableExtractor()
    with open("./tests/data/15601346.html", "r") as fobj:
        html = fobj.read()
        tables = extractor.extract_tables(html)
        assert len(tables) == 1
        assert len(tables[0]["caption"]) > 0
        assert len(tables[0]["footnote"]) > 0

        title = extractor.extract_title(html)
        assert title is not None

def test_HtmlTableExtractor_18782787():
    extractor = HtmlTableExtractor()
    with open("./tests/data/18782787.html", "r") as fobj:
        html = fobj.read()
        tables = extractor.extract_tables(html)
        assert len(tables) > 0

        title = extractor.extract_title(html)
        assert title is not None

    for table in tables:
        md_table = dataframe_to_markdown(table["table"])
        assert len(md_table) > 0


def test_HtmlTableExtractor_xml_3071444():
    extractor = HtmlTableExtractor()
    with open("./data/20260224/incr.2024-12-17/PMC003xxxxxx/PMC3071444.xml", "r") as fobj:
        xml = fobj.read()
        tables = extractor.extract_tables(xml)
        assert len(tables) == 2
        assert len(tables[0]["caption"]) > 0
        assert tables[0]["table"] is not None

        title = extractor.extract_title(xml)
        assert title == "Clinical Utility of Real-Time Fusion Guidance for Biopsy and Ablation"

        abstract = extractor.extract_abstract(xml)
        assert abstract is not None
        assert len(abstract) > 0

        sections = extractor.extract_sections(xml)
        assert sections is not None
        assert len(sections) > 0
        assert sections[0]["section"] == "Abstract"


def test_HtmlTableExtractor_xml_3063650_table_footnote():
    extractor = HtmlTableExtractor()
    with open("./data/20260224/incr.2024-12-17/PMC003xxxxxx/PMC3063650.xml", "r") as fobj:
        xml = fobj.read()
        tables = extractor.extract_tables(xml)
        assert len(tables) > 0
        assert any(len(t.get("footnote", "")) > 0 for t in tables)
