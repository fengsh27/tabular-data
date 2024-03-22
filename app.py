import streamlit as st
import logging
import os
from dotenv import load_dotenv

from components.main_tab import main_tab
from components.html_table_tab import html_tab
from extractor.stampers import ArticleStamper

load_dotenv()
output_folder = os.environ.get("TMP_FOLDER", "./tmp")
stamper_enabled = os.environ.get("LOG_ARTICLE", "FALSE") == "TRUE"
stamper = ArticleStamper(output_folder, stamper_enabled)
def initialize():
    # prepare logger
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler("./logs/app.log")
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

tab1, tab2 = st.tabs(["Extract Tabula Data", "Extract html table"])

with tab1:
    main_tab(stamper)
with tab2:
    html_tab(stamper)