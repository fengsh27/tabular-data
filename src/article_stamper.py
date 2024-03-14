
from typing import Optional, Dict, List, Any
import os
from os import path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Stamper(object):
    def __init__(self, pmid: str):
        self.pmid = pmid

    def log_error(self, msg):
        pass
    def log_info(self, msg):
        pass
    def output_prompts(self, prompts: Any):
        pass
    def output_html(self, html_content: Any):
        pass
    def output_result(self, result: str):
        pass

class AricleStamper(Stamper):
    logs_file = "logs.log"
    
    def __init__(self, pmid: str, output_folder: Optional[str] = "./tmp"):
        super().__init__(pmid)
        self.enabled = os.environ.get("LOG_ARTICLE", "FALSE") == "TRUE"
        self.output_folder = output_folder
        self.pmid_folder = path.join(self.output_folder, self.pmid)
        
        # make sure {pmid} folder exists
        self._mk_pmid_dir()
    
    def log_error(self, msg: str):
        self._log_message("Error", msg)

    def log_info(self, msg: str):
        self._log_message("Info", msg)

    @staticmethod
    def _now_string(include_ms: Optional[bool]=False, in_filename: Optional[bool]=False):
        now = datetime.now()
        if not in_filename:
            return now.strftime("%m/%d/%Y, %H:%M:%S:%f") if include_ms else now.strftime("%m/%d/%Y, %H:%M:%S")
        else:
            return now.strftime("%Y.%m.%d_%H.%M.%S.%f") if include_ms else now.strftime("%Y.%m.%d_%H.%M.%S")

    def _log_message(self, level:str, msg: str):
        tm_str = AricleStamper._now_string(True)
        msg = f"[{level}] {tm_str} = {msg}"
        self._write_message(AricleStamper.logs_file, msg)

    def _mk_pmid_dir(self):
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

    def output_prompts(self, prompts: List[Dict[str, str]]):
        msg = ""
        for prmpt in prompts:
            the_msg = f'["role": {prmpt["role"]}, "content": {prmpt["content"]}]'
            msg += the_msg
            msg += "\n"
        now_str = AricleStamper._now_string(in_filename=True, include_ms=False)
        self._write_message(f"{self.pmid}-{now_str}.prompts", msg, False)

    def output_html(self, html_content: str):
        now_str = AricleStamper._now_string(in_filename=True, include_ms=False)
        self._write_message(f"{self.pmid}-{now_str}.html", html_content, False)
    
    def output_result(self, result: str):
        now_str = AricleStamper._now_string(in_filename=True, include_ms=False)
        self._write_message(f"{self.pmid}-{now_str}.result", result)

