import base64
import logging.handlers
import streamlit as st
import logging
import os
from dotenv import load_dotenv
load_dotenv()

from components.main_tab import main_tab
from components.html_table_tab import html_tab
from extractor.stampers import ArticleStamper
from version import __version__

def initialize():
    # prepare logger
    # logging.basicConfig(level=logging.INFO)
    logs_folder = os.environ.get("LOGS_FOLDER", "./logs")
    logs_file = os.path.join(logs_folder, "app.log")
    
    # Root logger configuration (optional)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Silence noisy libraries

    # extractor logger
    extractor_logger = logging.getLogger("extractor")
    extractor_logger.setLevel(logging.INFO)
    extractor_logger.handlers.clear()

    # components logger
    components_logger = logging.getLogger("components")
    components_logger.setLevel(logging.INFO)
    components_logger.handlers.clear()

    # app logger
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)
    app_logger.handlers.clear()

    file_handler = logging.handlers.RotatingFileHandler(logs_file)
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        # datefmt="%Y-%m-%d %H:%M:%S,uuu"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    extractor_logger.addHandler(file_handler)
    extractor_logger.addHandler(stream_handler)
    components_logger.addHandler(file_handler)
    components_logger.addHandler(stream_handler)
    app_logger.addHandler(file_handler)
    app_logger.addHandler(stream_handler)

    # Prevent propagation to root logger
    extractor_logger.propagate = False
    components_logger.propagate = False
    app_logger.propagate = False
    return app_logger

logger = initialize()

ss = st.session_state
st.set_page_config(layout="wide", page_title="Curation Tool", page_icon="./images/favicon.png")

st.markdown(
    """<div style="display: flex;flex-direction: row; align-items: flex-end"><a href="https://mprint.org/index.html"><img src="data:image/png;base64,{}" width="400px"></a><p>Curation Tool v{}</p></div> """.format(
    base64.b64encode(open("./images/mprint-logo.png", "rb").read()).decode(), __version__),
    unsafe_allow_html=True,
)

# st.markdown('[<img src="images/copper-logo.png">](https://mprint.org/index.html)')
# st.image("./images/copper-logo.png", width=500)
# tab1, tab2 = st.tabs(["Extract From Table Data", "Extract From The Full-text"])

# with tab1:
main_tab()
# with tab2:
#     pass
    # html_tab()