
from bs4 import BeautifulSoup, Tag
from typing import Optional
from extractor.utils import convert_html_table_to_dataframe

class HtmlTableParser(object):
    MAX_LEVEL = 3
    CAPTION_CANDIDATES = ["caption", "captions", "title"]
    FOOTNOTE_CANDIDATES = ["note", "legend", "description", "foot"]
    def __init__(self):
        pass

    def _get_caption_or_footnote_text(self, tag:Tag)->str:
        text = tag.text
        if text is not None and len(text) > 0:
            return text
        children = list(tag.descendants)
        if len(children) == 0:
            return text
        for child in children:
            the_text = child.text
            text += the_text
        return text
    @staticmethod
    def _is_caption_in_text(text):
        for cap in HtmlTableParser.CAPTION_CANDIDATES:
            if cap in text:
                return True
        return False
    @staticmethod
    def _is_footnote_in_text(text):
        for foot in HtmlTableParser.FOOTNOTE_CANDIDATES:
            if foot in text:
                return True
        return False
    def _find_caption_and_footnote_recursively(
            self, 
            parent_tag: Tag | None, 
            level: int, 
            found_caption: Optional[bool]=False,
            found_footnote: Optional[bool]=False
        )->tuple[str, str, Tag | None]:
        if parent_tag is None:
            return "", "", None
        if level > HtmlTableParser.MAX_LEVEL:
            return "", "", None
        children = parent_tag.children
        caption = None
        footnote = None
        for child in children:
            if not hasattr(child, "attrs"):
                continue
            classes = child.attrs.get("class")
            if classes is None:
                continue
            if not isinstance(classes, str):
                try:
                    classes = ' '.join(classes)
                except:
                    continue
            if not found_caption and HtmlTableParser._is_caption_in_text(classes):
                caption = self._get_caption_or_footnote_text(child)
                found_caption = True
            if not found_footnote and HtmlTableParser._is_footnote_in_text(classes):
                footnote = self._get_caption_or_footnote_text(child)
                found_footnote = True
        if found_caption and found_footnote:
            return caption, footnote, parent_tag
        if not found_caption and not found_footnote:
            return self._find_caption_and_footnote_recursively(
                parent_tag.parent, level+1, found_caption, found_footnote
            )
        if not found_caption:
            caption, _, further_parent_tag = self._find_caption_and_footnote_recursively(
                parent_tag.parent, level+1, found_caption, found_footnote
            )
            final_parent_tag = further_parent_tag if further_parent_tag is not None \
                and (caption is not None and len(caption) > 0) else parent_tag
            return caption, footnote, final_parent_tag
        if not found_footnote:
            _, footnote, further_parent_tag = self._find_caption_and_footnote_recursively(
                parent_tag.parent, level+1, found_caption, found_footnote
            )
            final_parent_tag = further_parent_tag if further_parent_tag is not None \
                and (footnote is not None and len(footnote) > 0) else parent_tag
            return caption, footnote, final_parent_tag

    def _find_caption_and_footnote(self, table_tag: Tag):
        return self._find_caption_and_footnote_recursively(table_tag.parent, 1)
    def extract_tables(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        tags = soup.select("table")
        tables = []
        for tag in tags:
            strTag = str(tag)
            table = convert_html_table_to_dataframe(strTag)
            if table is None:
                continue
            caption, footnote, parent_tag = self._find_caption_and_footnote(tag)
            parent_tag = parent_tag if parent_tag is not None else tag
            tables.append({
                "caption": caption if caption is not None else "",
                "footnote": footnote if footnote is not None else "",
                "table": table,
                "raw_tag": str(parent_tag),
            })
        return tables

class PMCHtmlTableParser(object):
    def __init__(self):
        pass
    def extract_tables(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        tags = soup.select("div.table-wrap.anchored.whole_rhythm")
        tables = []
        for tag in tags:
            tbl_soup = BeautifulSoup(str(tag), "html.parser")
            caption = tbl_soup.select("div.caption")
            caption = caption[0].text if len(caption) > 0 else ""
            table = tbl_soup.select("div.xtable")
            table = str(table[0]) if len(table) > 0 else ""
            table = convert_html_table_to_dataframe(table)
            footnote = tbl_soup.select("div.tblwrap-foot")
            footnote = footnote[0].text if len(footnote) > 0 else ""
            tables.append({
                "caption": caption, 
                "table": table, 
                "footnote": footnote,
                "raw_tag": str(tag),
            })
    
        return tables

class HtmlTableExtractor(object):
    def __init__(self):
        self.parsers = [
            PMCHtmlTableParser(),
            HtmlTableParser(),    
        ]

    def extract_tables(self, html: str):
        for parser in self.parsers:
            tables = parser.extract_tables(html)
            if tables and len(tables) > 0:
                return tables
        return []

