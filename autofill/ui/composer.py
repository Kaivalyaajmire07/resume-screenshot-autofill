"""ChatGPT/Claude-style screenshot composer: paste/upload -> thumbnail gallery -> instruction -> Extract."""
from __future__ import annotations

import streamlit as st

from .. import config
from ..attachments import AttachmentQueue

try:  # optional dependency: the uploader still works without it
    from streamlit_paste_button import paste_image_button
except Exception:  # pragma: no cover
    paste_image_button = None

THUMBS_PER_ROW = 5


def _note(msg: str) -> None:
    st.session_state.setdefault("composer_notes", []).append(msg)


def _ingest(queue: AttachmentQueue) -> None:
    """Move newly pasted/uploaded images into the queue, then reset the widgets and rerun."""
    changed = False
    gen_paste = st.session_state.setdefault("paste_gen", 0)
    gen_up = st.session_state.setdefault("uploader_gen", 0)

    if paste_image_button is not None:
        from io import BytesIO
        from datetime import datetime

        result = paste_image_button(label="📋  Paste screenshot", key=f"paste_{gen_paste}", errors="ignore")
        if result.image_data is not None:
            buf = BytesIO()
            result.image_data.save(buf, format="PNG")
            ok, msg = queue.add(f"pasted_{datetime.now().strftime('%H%M%S_%f')}.png", buf.getvalue())
            if not ok:
                _note(msg)
            st.session_state["paste_gen"] = gen_paste + 1
            changed = True

    files = st.file_uploader(
        "Attach screenshots or PDF resumes (drag & drop or browse — select many at once)",
        type=list(config.ALLOWED_EXTENSIONS), accept_multiple_files=True, key=f"uploader_{gen_up}")
    if files:
        for f in files:
            ok, msg = queue.add(f.name, f.getvalue())
            if not ok:
                _note(msg)
        st.session_state["uploader_gen"] = gen_up + 1
        changed = True
    if changed:
        st.rerun()


def _gallery(queue: AttachmentQueue) -> None:
    items = list(queue.items)
    for start in range(0, len(items), THUMBS_PER_ROW):
        cols = st.columns(THUMBS_PER_ROW)
        for col, item in zip(cols, items[start:start + THUMBS_PER_ROW]):
            n = queue.items.index(item) + 1
            with col:
                st.image(item.thumb)
                badge = " <span class='thumb-badge'>cropped</span>" if item.cropped else ""
                st.markdown(f"<div class='thumb-cap'>{n}. {item.name}{badge}</div>", unsafe_allow_html=True)
                b1, b2, b3 = st.columns(3)
                if b1.button("✕", key=f"rm_{item.id}", help="Remove"):
                    queue.remove(item.id)
                    st.rerun()
                if b2.button("◀", key=f"up_{item.id}", help="Move earlier", disabled=n == 1):
                    queue.move(item.id, -1)
                    st.rerun()
                if b3.button("▶", key=f"dn_{item.id}", help="Move later", disabled=n == len(queue)):
                    queue.move(item.id, +1)
                    st.rerun()


def _crop_tool(queue: AttachmentQueue) -> None:
    with st.expander("✂️ Crop a screenshot before extraction"):
        item_id = st.selectbox("Screenshot / PDF", [i.id for i in queue.items],
                               format_func=lambda i: f"{[x.id for x in queue.items].index(i) + 1}. {queue.get(i).name}",
                               key="crop_item")
        item = queue.get(item_id)
        if item is None:
            return
        if item.is_pdf:
            st.caption("PDFs are read as they are; cropping is only for screenshots.")
            return
        c1, c2 = st.columns(2)
        left = c1.slider("Left (%)", 0, 90, 0, key=f"cl_{item_id}")
        top = c2.slider("Top (%)", 0, 90, 0, key=f"ct_{item_id}")
        c3, c4 = st.columns(2)
        width = c3.slider("Width (%)", 10, 100 - left, 100 - left, key=f"cw_{item_id}_{left}") if left < 90 else 10
        height = c4.slider("Height (%)", 10, 100 - top, 100 - top, key=f"ch_{item_id}_{top}") if top < 90 else 10
        from ..imaging import crop_percent

        st.image(crop_percent(item.original, left, top, width, height), caption="Crop preview")
        a, b = st.columns(2)
        if a.button("Apply crop", key=f"apply_{item_id}"):
            queue.apply_crop(item_id, left, top, width, height)
            st.rerun()
        if b.button("Reset crop", key=f"reset_{item_id}", disabled=not item.cropped):
            queue.reset_crop(item_id)
            st.rerun()


def render(queue: AttachmentQueue) -> tuple[bool, str]:
    """Draw the composer. Returns (extract_clicked, instruction_text)."""
    with st.container(key="composer"):
        _ingest(queue)
        notes = st.session_state.pop("composer_notes", [])
        for n in notes[:6]:
            st.warning(n)
        if queue.items:
            st.markdown(f"**{len(queue)} attachment(s)** · {config.MAX_UPLOADS} max per candidate run")
            _gallery(queue)
            _crop_tool(queue)
        instruction = st.text_area(
            "Ask or instruct (optional)", height=80, key="resume_user_instruction",
            placeholder="Optional note saved with this run, e.g. “Candidate for Python Developer opening”.")
        left, right = st.columns([4, 1])
        clicked = left.button("✨ Extract details", type="primary", disabled=not queue.items, key="extract")
        if right.button("Clear all", disabled=not queue.items, key="clear_all"):
            queue.clear()
            st.rerun()
    return clicked, instruction
