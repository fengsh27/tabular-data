import base64
import logging.handlers
import streamlit as st
import logging
from dotenv import load_dotenv

load_dotenv()

from components.main_tab import main_tab
from extractor.log_utils import initialize_logger

from version import __version__

app_logger = initialize_logger(
    log_file="app.log",
    app_log_name="app",
    app_log_level=logging.INFO,
    log_entries={
        "extractor": logging.INFO,
        "components": logging.INFO,
    },
)

ss = st.session_state
st.set_page_config(
    layout="wide", page_title="Curation Tool", page_icon="./images/favicon.png"
)

st.markdown(
    """<div style="display: flex;flex-direction: row; align-items: flex-end"><a href="https://mprint.org/index.html"><img src="data:image/png;base64,{}" width="400px"></a><p>Curation Tool v{}</p></div> """.format(
        base64.b64encode(open("./images/mprint-logo.png", "rb").read()).decode(),
        __version__,
    ),
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
