"""Dark theme polish. Base colours come from .streamlit/config.toml; this adds the composer look."""
import streamlit as st

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h1 { letter-spacing: -0.02em; }
.app-sub { color: #9aa4b2; margin: -0.6rem 0 1.1rem 0; font-size: 0.95rem; }

/* ChatGPT/Claude-style composer */
.st-key-composer {
  border: 1px solid #2d3445; border-radius: 24px; padding: 18px 20px 14px 20px;
  background: linear-gradient(180deg, #171b24 0%, #131720 100%);
  box-shadow: 0 6px 28px rgba(0,0,0,.35);
}
.st-key-composer:focus-within { border-color: #6b86ff; }
.st-key-composer textarea { border-radius: 16px !important; }
.thumb-cap { font-size: .72rem; color: #9aa4b2; white-space: nowrap; overflow: hidden;
             text-overflow: ellipsis; margin-top: -.35rem; }
.thumb-badge { font-size: .68rem; color: #7c9cff; }
div[data-testid="stImage"] img { border-radius: 12px; border: 1px solid #2d3445; }

/* review form */
.pill { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem;
        border: 1px solid #2d3445; margin-right: 6px; color: #c7cdd8; }
.pill.warn { border-color: #b7791f; color: #f6ad55; }
.pill.ok { border-color: #2f855a; color: #68d391; }
.stButton > button[kind="primary"] { border-radius: 14px; font-weight: 600; }
.stButton > button { border-radius: 12px; }
.st-key-extract button { width: 100%; }
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
