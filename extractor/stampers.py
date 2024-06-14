
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
        self.name = None
        self.pmid_folder = None
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
        self._ensure_pmid_folder_exist()
        msg = ""
        for prmpt in prompts:
            the_msg = f'["role": {prmpt["role"]}, "content": {prmpt["content"]}]'
            msg += the_msg
            msg += "\n"
        now_str = ArticleStamper._now_string(in_filename=True, include_ms=False)
        self._write_message(f"{self.name}-{now_str}.prompts", msg, False)

    def output_html(self, html_content: str):
        self._ensure_pmid_folder_exist()
        now_str = ArticleStamper._now_string(in_filename=True, include_ms=False)
        self._write_message(f"{self.name}-{now_str}.html", html_content, False)
    
    def output_result(self, result: str):
        self._ensure_pmid_folder_exist()
        now_str = ArticleStamper._now_string(in_filename=True, include_ms=False)
        self._write_message(f"{self.name}-{now_str}.result", result)

    def output_screenshot(self, png: bytearray):
        self._ensure_pmid_folder_exist()
        now_str = ArticleStamper._now_string(in_filename=True, include_ms=False)
        self._write_binary_content(f"{self.name}-{now_str}.png", png)   

    def _ensure_pmid_folder_exist(self):
        if self.name is None:
            self.name = self.pmid
        try:
            if self.pmid_folder is None:
                self.pmid_folder = path.join(self.output_folder, self.name)
            if path.exists(self.pmid_folder):
                return
            os.mkdir(self.pmid_folder)
        except Exception as e:
            logger.error(e)
            logger.error(f"pmid: {self._pmid}, name: {self.name}, pmid_folder: {self.pmid_folder}, output_folder: {self.output_folder}")
            if self.name is None:
                self.name = "unknown"
            if self.output_folder is None:
                self.output_folder = "/tmp"
            self.pmid_folder = "/tmp/unknown"
            if not path.exists(self.pmid_folder):
                os.mkdir(self.pmid_folder)


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
        self._ensure_pmid_folder_exist()

    def _write_message(self, fn:str, msg: str, is_append=True):
        if not self.enabled:
            print(msg)
            return
        self._ensure_pmid_folder_exist()
        file_path = path.join(self.pmid_folder, fn)
        with open(file_path, "a+" if is_append else "w+") as fobj:
            fobj.write(msg)

    def _write_binary_content(self, fn: str, content: bytearray):
        if not self.enabled:
            return
        self._ensure_pmid_folder_exist()
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
