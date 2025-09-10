import asyncio
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
    f"""
<div style="display: flex; flex-direction: row; align-items: flex-start; gap: 20px; margin-bottom: 1.5rem;">
    <a href="https://mprint.org/index.html" target="_blank" style="margin: 0; padding: 0;">
        <img src="data:image/png;base64,{base64.b64encode(open('./images/mprint-logo.png', 'rb').read()).decode()}" 
             width="250px" style="margin: 0; padding: 0;">
    </a>
    <div style="font-size: 1.6rem; font-weight: 600; line-height: 1; transform: translateY(50px); margin: 0; padding: 0;">
        Tabular Curation Tool
        <span style="font-size: 0.9rem; color: gray;"> v{__version__}</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# st.markdown('[<img src="images/copper-logo.png">](https://mprint.org/index.html)')
# st.image("./images/copper-logo.png", width=500)
# tab1, tab2 = st.tabs(["Extract From Table Data", "Extract From The Full-text"])

# with tab1:
asyncio.run(main_tab())
# with tab2:
#     pass
# html_tab()
