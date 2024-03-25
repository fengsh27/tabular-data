
from typing import Optional, Dict, List, Any
import os
from os import path
import logging
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class Stamper(ABC):
    def __init__(self, output_folder, enabled: Optional[bool]=False):
        self.enabled = enabled
        self.output_folder = output_folder
        self._pmid = None
    @property
    def pmid(self):
        return self._pmid
    
    @pmid.setter
    def pmid(self, pmid: str):
        self._pmid = pmid
        self._mk_pmid_dir()

    @abstractmethod
    def output_prompts(self, prompts: Any):
        pass
    @abstractmethod
    def output_html(self, html: Any):
        pass
    @abstractmethod
    def output_result(self, result: str):
        pass
    @abstractmethod
    def output_screenshot(self, png: bytearray):
        pass
    @abstractmethod
    def _mk_pmid_dir():
        pass

class ArticleStamper(Stamper):
    def __init__(self, output_folder, enabled: Optional[bool]=False):
        super().__init__(output_folder, enabled)

    def output_prompts(self, prompts: List[Dict[str, str]]):
        msg = ""
        for prmpt in prompts:
            the_msg = f'["role": {prmpt["role"]}, "content": {prmpt["content"]}]'
            msg += the_msg
            msg += "\n"
        now_str = ArticleStamper._now_string(in_filename=True, include_ms=False)
        self._write_message(f"{self.name}-{now_str}.prompts", msg, False)

    def output_html(self, html_content: str):
        now_str = ArticleStamper._now_string(in_filename=True, include_ms=False)
        self._write_message(f"{self.name}-{now_str}.html", html_content, False)
    
    def output_result(self, result: str):
        now_str = ArticleStamper._now_string(in_filename=True, include_ms=False)
        self._write_message(f"{self.name}-{now_str}.result", result)

    def output_screenshot(self, png: bytearray):
        now_str = ArticleStamper._now_string(in_filename=True, include_ms=False)
        self._write_binary_content(f"{self.name}-{now_str}.png", png)   

    def _mk_pmid_dir(self):
        if not self.enabled:
            return
        name = self.pmid
        if name.startswith("http://") or name.startswith("https://"):
            # pmid is url
            ix = name.rfind("/")
            if ix > 0:
                name = name[(ix+1):]
            ix = name.find("?")
            if ix > 0:
                name = name[:ix]
        self.name = name
        self.pmid_folder = path.join(self.output_folder, name)
        try:
            if path.exists(self.pmid_folder):
                return
            os.mkdir(self.pmid_folder)
        except Exception as e:
            logger.error(e)

    def _write_message(self, fn:str, msg: str, is_append=True):
        if not self.enabled:
            print(msg)
            return
        file_path = path.join(self.pmid_folder, fn)
        with open(file_path, "a+" if is_append else "w+") as fobj:
            fobj.write(msg)

    def _write_binary_content(self, fn: str, content: bytearray):
        if not self.enabled:
            return
        file_path = path.join(self.pmid_folder, fn)
        with open(file_path, "wb") as fobj:
            fobj.write(content)

    @staticmethod
    def _now_string(include_ms: Optional[bool]=False, in_filename: Optional[bool]=False):
        now = datetime.now()
        if not in_filename:
            return now.strftime("%m/%d/%Y, %H:%M:%S:%f") if include_ms else now.strftime("%m/%d/%Y, %H:%M:%S")
        else:
            return now.strftime("%Y.%m.%d_%H.%M.%S.%f") if include_ms else now.strftime("%Y.%m.%d_%H.%M.%S")
