"""Resume Screenshot Autofill — Streamlit entry point.  Run:  streamlit run app.py"""
import os

import streamlit as st

from autofill import config
from autofill.attachments import AttachmentQueue
from autofill.db import SQLiteHistoryRepository
from autofill.pipeline import process
from autofill.retention import purge_expired
from autofill.ui import composer, history_view, review, styles

st.set_page_config(page_title="Resume Screenshot Autofill", page_icon="🧾", layout="wide")


@st.cache_resource(show_spinner=False)
def get_repo() -> SQLiteHistoryRepository:
    return SQLiteHistoryRepository(config.DB_PATH)


@st.cache_resource(show_spinner=False)
def start_ocr_warmup() -> bool:
    """Load the OCR models in the background once per server start."""
    import threading

    from autofill.ocr_engine import get_engine

    threading.Thread(target=get_engine().warm_up, daemon=True).start()
    return True


repo = get_repo()
if not os.environ.get("RESUME_NO_WARMUP"):
    start_ocr_warmup()
purge_expired(repo, min_interval=60)  # at most once a minute, plus once at start-up
styles.apply()

st.title("🧾 Resume Screenshot Autofill")
st.markdown("<div class='app-sub'>Local OCR · nothing leaves this machine · editable fields · "
            "today's history only</div>", unsafe_allow_html=True)

if "queue" not in st.session_state:
    st.session_state["queue"] = AttachmentQueue()
queue: AttachmentQueue = st.session_state["queue"]

tab_process, tab_history = st.tabs(["➕ Process screenshots", "🕘 Today's history"])

with tab_process:
    clicked, instruction = composer.render(queue)
    if clicked:
        bar = st.progress(0.0, text="Preparing screenshots…")
        with st.spinner("Reading screenshots… the first run loads the OCR models and can take a while."):
            result = process(list(queue.items),
                             progress=lambda done, total: bar.progress(done / total, text=f"Read {done} of {total}"))
        bar.empty()
        st.session_state.update(result=result, run_id=st.session_state.get("run_id", 0) + 1,
                                saved_id=None, screenshots=len(queue), instruction=instruction.strip())
        if not result.raw_text.strip() or not result.lines:
            st.error("No text was extracted. Try clearer or larger screenshots — see the warnings below.")
    review.render(repo)

with tab_history:
    history_view.render(repo)
