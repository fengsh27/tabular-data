import base64
import streamlit as st
import logging
import os
from dotenv import load_dotenv
load_dotenv()

from components.main_tab import main_tab
from components.html_table_tab import html_tab
from extractor.stampers import ArticleStamper

__version__ = "0.1.37"

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
st.set_page_config(layout="wide", page_title="Curation Tool", page_icon="./images/favicon.png")

st.markdown(
    """<div style="display: flex;flex-direction: row; align-items: flex-end"><a href="https://mprint.org/index.html"><img src="data:image/png;base64,{}" width="400px"></a><p>Curation Tool v{}</p></div> """.format(
    base64.b64encode(open("./images/mprint-logo.png", "rb").read()).decode(), __version__),
    unsafe_allow_html=True,
)

# st.markdown('[<img src="images/copper-logo.png">](https://mprint.org/index.html)')
# st.image("./images/copper-logo.png", width=500)
tab1, tab2 = st.tabs(["Extract From Table Data", "Extract From The Full-text"])

with tab1:
    main_tab()
with tab2:
    pass
    # html_tab()