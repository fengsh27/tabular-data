import streamlit as st
import logging
import os
from dotenv import load_dotenv
load_dotenv()

from components.main_tab import main_tab
from components.html_table_tab import html_tab
from extractor.stampers import ArticleStamper

output_folder = os.environ.get("TEMP_FOLDER", "./tmp")
stamper_enabled = os.environ.get("LOG_ARTICLE", "FALSE") == "TRUE"
stamper = ArticleStamper(output_folder, stamper_enabled)
def initialize():
    # prepare logger
    logging.basicConfig(level=logging.INFO)
    logs_folder = os.environ.get("LOGS_FOLDER", "./logs")
    logs_file = os.path.join(logs_folder, "app.log")
    file_handler = logging.FileHandler(logs_file)
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        # datefmt="%Y-%m-%d %H:%M:%S,uuu"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

initialize()
logger = logging.getLogger(__name__)

ss = st.session_state
st.set_page_config(layout="wide")

tab1, tab2 = st.tabs(["Extract From Table Data", "Extract From The Full-text"])

with tab1:
    main_tab(stamper)
with tab2:
    pass
    # html_tab(stamper)