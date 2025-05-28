from bs4 import BeautifulSoup, Tag
from typing import Callable, Optional
import pandas as pd
from TabFuncFlow.utils.table_utils import html_table_to_markdown, dataframe_to_markdown
from extractor.utils import convert_html_table_to_dataframe
from typing import List, Optional, Dict


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

    def _find_first_occurrence(self,
                               soup: BeautifulSoup,
                               keywords: List[str]) -> Optional[Tag]:
        """
        Yichuan 0528
        """
        kw_lower = [k.lower() for k in keywords]
        valid_tags = {"section", "div", "article", "main", "h1", "h2", "h3", "p", "ul", "ol", "table", "article"}

        for el in soup.find_all(True):
            if el.name not in valid_tags:
                continue
            if any(kw in cls.lower() for cls in el.get("class", []) for kw in kw_lower):
                return el
            if any(kw in (el.get("id", "").lower()) for kw in kw_lower):
                return el
            direct = ''.join(el.find_all(string=True, recursive=False)).strip().lower()
            if any(kw in direct for kw in kw_lower):
                return el
        return None

    def extract_abstract(self, html: str):
        """
        Yichuan 0528
        """
        sections = self.extract_sections(html)
        for section in sections:
            if "abstract" in section["section"].lower():
                return section["content"].replace("\n", " ")
        return (sections[0]["section"] + "\n" + sections[0]["content"]).replace("\n", " ") + "\n......" or None

    def extract_sections(self, html: str):
        """
        Yichuan 0528
        Generic section extraction for main body content (non-PMC).
        Returns [{'section': ..., 'content': ...}, ...]
        """
        stop_sections = [
                "reference", "references",
                "acknowledgement", "acknowledgment",
                "acknowledgements", "acknowledgments",
                "supplementary", "supplements"
        ]

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a"):
            a.decompose()

        start = self._find_first_occurrence(soup, ["abstract"])
        if not start:
            return []

        sections: List[Dict[str, str]] = []
        current: Optional[Dict[str, str]] = None
        seen_global = set()
        heading_tags = ["h1", "h2", "h3", "h4"]
        block_tags = ["p", "ul", "ol", "div", "section", "article"]

        for el in start.find_all_next():
            # ── 1. Handle section headings ───────────────────────────────
            if el.name in heading_tags:
                h_raw = el.get_text(strip=True)
                h_low = h_raw.lower()

                # On encountering References/Acknowledgements → finalize and exit
                if any(kw in h_low for kw in stop_sections):
                    if current and current["content"].strip():
                        lines = list(dict.fromkeys(current["content"].splitlines()))
                        current["content"] = "\n".join(lines).strip()
                        sections.append(current)
                    break

                # Normal new heading
                if current and current["content"].strip():
                    lines = list(dict.fromkeys(current["content"].splitlines()))
                    current["content"] = "\n".join(lines).strip()
                    sections.append(current)
                current = {"section": h_raw, "content": ""}
                continue

            # ── 2. Collect main body text ───────────────────────────────
            if current is None:
                continue  # Not yet in the first main body section

            if el.name == "table":
                try:
                    df = convert_html_table_to_dataframe(str(el))
                    # I use this one instead of the custom html_table_to_dataframe implementation,
                    # because it uses StringIO and is likely more robust.
                    # That said, the previous custom version hasn’t caused any major issues so far,
                    # so I’m not eager to change it either.  - Yichuan 0528
                    markdown = dataframe_to_markdown(df)
                    if markdown and markdown not in seen_global:
                        current["content"] += markdown + "\n"
                        seen_global.add(markdown)
                    continue
                except Exception:
                    pass

            if el.name in block_tags:
                txt = el.get_text(separator="\n", strip=True)
                if txt and txt not in seen_global:
                    current["content"] += txt + "\n"
                    seen_global.add(txt)

        # Document ended but still has current section
        if current and current["content"].strip():
            lines = list(dict.fromkeys(current["content"].splitlines()))
            current["content"] = "\n".join(lines).strip()
            sections.append(current)

        return sections


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

    def extract_abstract(self, html: str):
        """
        Yichuan 0501
        """
        soup = BeautifulSoup(html, "html.parser")

        # Find a heading tag that contains the word "abstract" (case-insensitive)
        abstract_heading = soup.find(
            lambda tag: tag.name in ["h2", "h3"] and "abstract" in tag.get_text(strip=True).lower()
        )

        if not abstract_heading:
            return None  # No abstract heading found

        # Find the parent <section> or container that wraps the abstract
        abstract_section = abstract_heading.find_parent("section")
        if not abstract_section:
            return None  # No wrapping section found

        # Extract all <p> tags (paragraphs) under the abstract section
        abstract_paragraphs = abstract_section.find_all("p")

        # Combine the text from each paragraph
        abstract_text = "\n".join(p.get_text(separator=" ", strip=True) for p in abstract_paragraphs)

        return abstract_text.strip()

    def extract_sections(self, html: str):
        """
        Yichuan 0505
        Extracts sections (h2/h3) and content between 'Abstract' and 'References' headings.
        Include tables
        """
        stop_sections = [
            "reference", "acknowledgement", "acknowledgment", "supplementary",
            "references", "acknowledgements", "acknowledgments", "supplements"
        ]
        soup = BeautifulSoup(html, "html.parser")
        body = soup.body
        if not body:
            return []

        heading_tags = ["h2", "h3"]
        sections = []
        current_section = None
        started = False

        for element in body.descendants:
            if isinstance(element, Tag):
                if element.name in heading_tags:
                    heading_text = element.get_text(strip=True).lower()

                    if not started and "abstract" in heading_text:
                        started = True
                        current_section = {
                            "section": element.get_text(strip=True),
                            "content": ""
                        }
                        continue

                    if started and any(
                            x in heading_text for x in stop_sections):
                        if current_section:
                            current_section["content"] = current_section["content"].strip()
                            sections.append(current_section)
                            current_section = None
                        break

                    if started:
                        if current_section:
                            current_section["content"] = current_section["content"].strip()
                            sections.append(current_section)
                        current_section = {
                            "section": element.get_text(strip=True),
                            "content": ""
                        }

                elif started and current_section:
                    # Tables: convert HTML
                    if element.name == "table" or (
                    "xtable" in element.get("class", []) if element.has_attr("class") else False):
                        current_section["content"] += html_table_to_markdown(str(element)) + "\n"

                    # Text elements: convert to plain text
                    elif element.name in ["p", "ul", "ol"]:
                        text = element.get_text(separator=" ", strip=True)
                        if text:
                            current_section["content"] += text + "\n"

        if current_section:
            current_section["content"] = current_section["content"].strip()
            sections.append(current_section)

        return sections


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

    def extract_abstract(self, html: str):
        """
        Yichuan 0501
        """
        for parser in self.parsers:
            title = parser.extract_abstract(html)
            if title is not None:
                return title

        return None

    def extract_sections(self, html: str):
        """
        Yichuan 0505
        """
        for parser in self.parsers:
            sections = parser.extract_sections(html)
            if sections is not None:
                return sections

        return None

    @staticmethod
    def convert_sections_to_chunks(sections, min_chunk_size=2048):
        """
        This method is currently unused, as processing the entire article as a whole gets better results.

        Converts structured sections into a list of text chunks, each at least `min_chunk_size` characters long.
        Filters out:
          1. Sections with empty or whitespace-only content
          2. Completely duplicate sections (same title and content)
        Ensures all chunks meet the minimum length requirement (except possibly the last one).
        Yichuan 0502
        """
        # Step 1: Remove empty and duplicate sections
        seen = set()
        filtered_sections = []
        for s in sections:
            title = s.get("section", "").strip()
            content = s.get("content", "").strip()
            if not content:
                continue  # Skip empty content
            key = (title, content)
            if key in seen:
                continue  # Skip exact duplicates
            seen.add(key)
            filtered_sections.append({"section": title, "content": content})

        # Step 2: Accumulate and chunk
        chunks = []
        buffer = ""

        for section in filtered_sections:
            complete_section = section["section"] + ":\n" + section["content"]
            buffer += "\n\n" + complete_section

            if len(buffer.strip()) >= min_chunk_size:
                chunks.append(buffer.strip())
                buffer = ""

        # Step 3: Handle remaining buffer
        if buffer.strip():
            if len(buffer.strip()) < min_chunk_size and chunks:
                chunks[-1] += "\n\n" + buffer.strip()
            else:
                chunks.append(buffer.strip())

        return chunks

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
