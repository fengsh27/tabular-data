from bs4 import BeautifulSoup, Tag
from typing import Callable, Optional
import pandas as pd

from extractor.utils import convert_html_table_to_dataframe

def get_tag_text(tag: Tag) -> str:
    text = tag.text
    text = text.strip()
    if text is not None and len(text) > 0:
        return text
    children = list(tag.descendants)
    if len(children) == 0:
        return text
    for child in children:
        the_text = child.text
        text += the_text
    return text

class HtmlTableParser(object):
    MAX_LEVEL = 3
    CAPTION_TAG_CANDIDATES = ["figcaption"]
    CAPTION_CANDIDATES = ["caption", "captions", "title"]
    FOOTNOTE_CANDIDATES = ["note", "legend", "description", "foot", "notes"]

    def __init__(self):
        pass

    def _get_caption_or_footnote_text(self, tag: Tag) -> str:
        return get_tag_text(tag)

    @staticmethod
    def _is_caption_in_text(text):
        for cap in HtmlTableParser.CAPTION_CANDIDATES:
            if cap in text:
                return True
        return False
    
    @staticmethod
    def _is_caption_by_tagname(tagname: str) -> bool:
        return tagname in HtmlTableParser.CAPTION_TAG_CANDIDATES

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
        found_caption: Optional[bool] = False,
        found_footnote: Optional[bool] = False,
    ) -> tuple[str, str, Tag | None]:
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
            if hasattr(child, "name") and HtmlTableParser._is_caption_by_tagname(tagname=child.name):
                caption = self._get_caption_or_footnote_text(child)
                found_caption = True
                continue
            classes = child.attrs.get("class")
            if classes is None:
                continue
            if not isinstance(classes, str):
                try:
                    classes = " ".join(classes)
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
                parent_tag.parent, level + 1, found_caption, found_footnote
            )
        if not found_caption:
            caption, _, further_parent_tag = (
                self._find_caption_and_footnote_recursively(
                    parent_tag.parent, level + 1, found_caption, found_footnote
                )
            )
            final_parent_tag = (
                further_parent_tag
                if further_parent_tag is not None
                and (caption is not None and len(caption) > 0)
                else parent_tag
            )
            return caption, footnote, final_parent_tag
        if not found_footnote:
            _, footnote, further_parent_tag = (
                self._find_caption_and_footnote_recursively(
                    parent_tag.parent, level + 1, found_caption, found_footnote
                )
            )
            final_parent_tag = (
                further_parent_tag
                if further_parent_tag is not None
                and (footnote is not None and len(footnote) > 0)
                else parent_tag
            )
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
            tables.append(
                {
                    "caption": caption if caption is not None else "",
                    "footnote": footnote if footnote is not None else "",
                    "table": table,
                    "raw_tag": str(parent_tag),
                }
            )
        return tables
    
    def _traverse_up(self, cur: Tag | None, level: int, max_level: int, check_cb: Callable):
        if cur is None:
            return False
        if level == max_level:
            return False
        res = check_cb(cur)
        return res if res else self._traverse_up(cur.parent, level+1, max_level, check_cb)
    
    def _traverse_down(self, cur: Tag | None, level: int, max_level: int, check_cb: Callable):
        if cur is None:
            return False
        if level == max_level:
            return False
        res = check_cb(cur)
        if res:
            return res
        try:
            for child in cur.children:
                res = self._traverse_down(child, level+1, max_level, check_cb)
                if res:
                    return res
        except AttributeError:
            return False
        return False
        

    def extract_title(self, html: str):
        def check_title_in_tag_classes(tag: Tag):
            if tag is None:
                return False
            try:
                if tag.attrs is None:
                    return False
            except AttributeError:
                return False
            classes = tag.attrs.get("class")
            if classes is None:
                return False
            if not isinstance(classes, str):
                try:
                    classes = " ".join(classes)
                except:
                    return False
            title_in_classes = "title" in classes
            if title_in_classes:
                return True
            id = tag.attrs.get('id')
            if id is not None and "title" in id.lower():
                return True
            else:
                return False
        
        soup = BeautifulSoup(html, "html.parser")
        tags = soup.select("h1")
        for tag in tags:
            if self._traverse_up(tag, 1, 5, check_title_in_tag_classes):
                return get_tag_text(tag)
            if self._traverse_down(tag, 1, 3, check_title_in_tag_classes):
                return get_tag_text(tag)
        
        return None

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
            tables.append(
                {
                    "caption": caption,
                    "table": table,
                    "footnote": footnote,
                    "raw_tag": str(tag),
                }
            )

        return tables

    def extract_title(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        tags = soup.select("hgroup h1")
        for tag in tags:
            text = get_tag_text(tag)
            if len(text.strip()) > 0:
                return text.strip()
        return None


class HtmlTableExtractor(object):
    def __init__(self):
        self.parsers = [
            PMCHtmlTableParser(),
            HtmlTableParser(),
        ]

    def extract_tables(self, html: str):
        tables = []
        for parser in self.parsers:
            tables = parser.extract_tables(html)
            if tables and len(tables) > 0:
                break

        tables = HtmlTableExtractor._remove_duplicate(tables)
        return tables
    
    def extract_title(self, html: str):
        for parser in self.parsers:
            title = parser.extract_title(html)
            if title is not None:
                return title
            
        return None


    @staticmethod
    def _tables_eq(tablel: dict, table2: dict) -> bool:
        df_table1: pd.DataFrame = tablel["table"]
        df_table2: pd.DataFrame = table2["table"]

        return df_table1.equals(df_table2)

    @staticmethod
    def _remove_duplicate(tables):
        res_tables = []
        for table in tables:
            if len(res_tables) == 0:
                res_tables.append(table)
                continue
            prev_table = res_tables[-1]
            if HtmlTableExtractor._tables_eq(prev_table, table):
                continue
            res_tables.append(table)

        return res_tables
