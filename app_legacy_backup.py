from datetime import date, datetime
from io import BytesIO
from pathlib import Path
import re
import sqlite3
import tempfile
import hashlib
import os

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
from streamlit_paste_button import paste_image_button

APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(__import__("os").environ.get("RESUME_DB_PATH", str(APP_DIR / "resume_history.db")))
MAX_UPLOADS = 30

st.set_page_config(page_title="Resume Screenshot Autofill", page_icon="🧾", layout="wide")

FIELDS = [
    ("first_name", "First Name"), ("middle_name", "Middle Name"), ("last_name", "Last Name"),
    ("date_of_birth", "Date of Birth"), ("gender", "Gender"), ("nationality", "Nationality"),
    ("marital_status", "Marital Status"), ("passport", "Passport"), ("hobbies", "Hobbies"),
    ("language_known", "Language Known"), ("address", "Address"), ("landmark", "Landmark"),
    ("city", "City"), ("state", "State"), ("pincode", "Pincode"), ("mobile", "Mobile"),
    ("email", "Email"), ("ssc_result", "SSC Result"), ("ssc_board", "SSC Board"),
    ("ssc_pass_year", "SSC Pass Year"), ("hsc_result", "HSC Result"), ("hsc_board", "HSC Board"),
    ("hsc_pass_year", "HSC Pass Year"), ("graduation_degree", "Graduation Degree"),
    ("graduation_result", "Graduation Result"), ("graduation_university", "Graduation University"),
    ("graduation_year", "Graduation Year"), ("post_graduation_degree", "Post Graduation Degree"),
    ("post_graduation_result", "Post Graduation Result"),
    ("post_graduation_university", "Post Graduation University"),
    ("post_graduation_year", "Post Graduation Year"), ("higher_level_education", "Higher Level Education"),
    ("objectives", "Objectives"), ("summary", "Summary"), ("key_skills", "Key Skills"),
    ("employment_details", "Employment Details"),
]
LABELS = dict(FIELDS)

def connect_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT NOT NULL,
            processed_at TEXT NOT NULL,
            source_count INTEGER NOT NULL,
            raw_text TEXT NOT NULL,
            fields_json TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

def purge_old_history(conn):
    # Daily-only history: retain records created on the current local date.
    today = date.today().isoformat()
    conn.execute("DELETE FROM history WHERE substr(processed_at, 1, 10) != ?", (today,))
    conn.commit()

def extract_text(uploaded_files):
    """Run local PaddleOCR on all uploaded screenshots; returns combined text and warnings."""
    from paddleocr import PaddleOCR
    # Lazy initialization means the OCR model is loaded only when extraction is requested.
    @st.cache_resource(show_spinner=False)
    def get_ocr():
        return PaddleOCR(use_textline_orientation=True, lang="en")
    ocr = get_ocr()
    chunks, warnings = [], []
    progress = st.progress(0, text="Preparing screenshots…")
    for idx, uploaded in enumerate(uploaded_files, start=1):
        try:
            image = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
            # Downscale oversized screenshots to reduce CPU inference time without
            # unnecessarily shrinking normal resume screenshots.
            max_edge = 1800
            if max(image.size) > max_edge:
                scale = max_edge / max(image.size)
                image = image.resize(
                    (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            image = ImageOps.autocontrast(image)
            image = ImageEnhance.Contrast(image).enhance(1.08)
            # On Windows, PaddleOCR cannot reopen a NamedTemporaryFile while Python still
            # holds it open. Close the file before OCR reads it, then remove it afterwards.
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                    image.save(tmp, format="PNG")
                result = ocr.predict(tmp_path)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
            lines = []
            for page in result or []:
                data = page.json if hasattr(page, "json") else page
                if callable(data):
                    data = data()
                if isinstance(data, dict):
                    res = data.get("res", data)
                    texts = res.get("rec_texts", []) if isinstance(res, dict) else []
                    lines.extend(str(t) for t in texts if str(t).strip())
                elif isinstance(page, (list, tuple)):
                    # Compatibility fallback for older PaddleOCR result structures.
                    for item in page:
                        try:
                            lines.append(str(item[1][0]))
                        except Exception:
                            pass
            if not lines:
                warnings.append(f"{uploaded.name}: OCR returned no readable text.")
            chunks.append(f"--- Screenshot {idx}: {uploaded.name} ---\n" + "\n".join(lines))
        except Exception as exc:
            warnings.append(f"{uploaded.name}: processing failed ({type(exc).__name__}: {exc}).")
        progress.progress(idx / len(uploaded_files), text=f"Processed {idx} of {len(uploaded_files)} screenshots")
    progress.empty()
    return "\n\n".join(chunks), warnings

def parse_fields(text):
    """Conservative baseline extraction. Unknown or unsupported fields remain blank."""
    values = {key: "" for key, _ in FIELDS}
    lines = [re.sub(r"\s+", " ", line).strip(" •\t") for line in text.splitlines()]
    lines = [line for line in lines if line and not line.startswith("--- Screenshot")]
    clean = "\n".join(lines)

    emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", clean)
    if emails:
        values["email"] = emails[0]
    phones = re.findall(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)", clean)
    for phone in phones:
        digits = re.sub(r"\D", "", phone)
        if 10 <= len(digits) <= 13:
            values["mobile"] = phone.strip()
            break

    # Name candidates are inferred from the first non-contact heading-like line.
    skip = re.compile(r"\b(resume|curriculum vitae|professional experience|education|key skills|objective|summary)\b", re.I)
    for line in lines[:20]:
        if "@" in line or re.search(r"\d{5,}", line) or skip.search(line):
            continue
        words = line.split()
        if 2 <= len(words) <= 5 and all(re.match(r"^[A-Za-z][A-Za-z.'’-]*$", w) for w in words):
            values["first_name"] = words[0]
            if len(words) > 2:
                values["middle_name"] = " ".join(words[1:-1])
            if len(words) > 1:
                values["last_name"] = words[-1]
            break

    def section(start, stops):
        start_idx = next((i for i, line in enumerate(lines) if re.search(start, line, re.I)), None)
        if start_idx is None:
            return ""
        out = []
        for line in lines[start_idx + 1:]:
            if any(re.search(stop, line, re.I) for stop in stops):
                break
            out.append(line)
        return "\n".join(out).strip()

    objective = section(r"^(career )?objective(s)?$", [r"^professional experience$", r"^experience$", r"^education$", r"^key skills$", r"^skills$"])
    summary = section(r"^(professional )?summary$", [r"^professional experience$", r"^experience$", r"^education$", r"^key skills$", r"^skills$"])
    skills = section(r"^(key )?skills$", [r"^education$", r"^professional experience$", r"^experience$", r"^projects?$"])
    experience = section(r"^(professional )?experience$", [r"^education$", r"^key skills$", r"^skills$", r"^projects?$"])
    if objective:
        values["objectives"] = objective
    if summary:
        values["summary"] = summary
    if skills:
        values["key_skills"] = "\n".join(f"• {x.lstrip('•-* ')}" for x in skills.splitlines())
    if experience:
        values["employment_details"] = experience

    # Conservative location parsing: use the address line only when it resembles a postal address.
    for line in lines[:12]:
        if re.search(r"\b(?:near|road|street|sector|extension|nagar|village|district|ghaziabad|noida)\b", line, re.I) and re.search(r"\d{6}\b", line):
            values["address"] = line
            pin = re.search(r"\b\d{6}\b", line)
            if pin:
                values["pincode"] = pin.group(0)
            break
    return values

def save_record(conn, fields, raw_text, source_count):
    name = " ".join(x for x in [fields.get("first_name", ""), fields.get("middle_name", ""), fields.get("last_name", "")] if x).strip()
    name = name or "Unknown Candidate"
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    import json
    conn.execute(
        "INSERT INTO history(candidate_name, processed_at, source_count, raw_text, fields_json) VALUES(?,?,?,?,?)",
        (name, stamp, source_count, raw_text, json.dumps(fields, ensure_ascii=False))
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

def get_record(conn, record_id):
    import json
    row = conn.execute("SELECT * FROM history WHERE id=?", (record_id,)).fetchone()
    if not row:
        return None
    result = dict(row)
    result["fields"] = json.loads(result.pop("fields_json"))
    return result

def update_record(conn, record_id, fields):
    import json
    name = " ".join(x for x in [fields.get("first_name", ""), fields.get("middle_name", ""), fields.get("last_name", "")] if x).strip() or "Unknown Candidate"
    conn.execute("UPDATE history SET candidate_name=?, fields_json=? WHERE id=?",
                 (name, json.dumps(fields, ensure_ascii=False), record_id))
    conn.commit()

conn = connect_db()
purge_old_history(conn)

st.title("🧾 Resume Screenshot Autofill")
st.caption("Local OCR • Editable fields • Today's history only")
st.info("Privacy and history: records are retained for today only. At the start of a new date, older history records are automatically deleted. Uploaded screenshots are processed temporarily and are not saved as image files.")

tab_process, tab_history = st.tabs(["➕ Process screenshots", "🕘 Today's history"])

with tab_process:
    st.subheader("Resume workspace")
    st.caption("Attach multiple screenshots, optionally crop them, add an instruction, then process the batch.")

    class ImageInput:
        def __init__(self, name, data):
            self.name = name
            self._data = data
        def getvalue(self):
            return self._data

    if "pasted_resume_images" not in st.session_state:
        st.session_state["pasted_resume_images"] = []
    if "pasted_resume_hashes" not in st.session_state:
        st.session_state["pasted_resume_hashes"] = set()
    if "cropped_resume_images" not in st.session_state:
        st.session_state["cropped_resume_images"] = {}
    if "cropped_uploaded_images" not in st.session_state:
        st.session_state["cropped_uploaded_images"] = {}

    st.markdown(
        """
        <style>
        .composer-shell {
            border: 1px solid rgba(128,128,128,.42);
            border-radius: 22px;
            padding: 14px 16px 12px 16px;
            background: rgba(128,128,128,.055);
            margin: 8px 0 12px 0;
        }
        </style>
        <div class="composer-shell"><b>Attachments</b><br/>Paste screenshots or select multiple image files. They stay attached until you clear them.</div>
        """,
        unsafe_allow_html=True,
    )

    paste_result = paste_image_button(
        label="＋ Paste screenshot from clipboard",
        key="resume_image_paste",
        errors="ignore",
    )
    if paste_result.image_data is not None:
        pasted_buffer = BytesIO()
        paste_result.image_data.save(pasted_buffer, format="PNG")
        pasted_bytes = pasted_buffer.getvalue()
        pasted_hash = hashlib.sha256(pasted_bytes).hexdigest()
        if pasted_hash not in st.session_state["pasted_resume_hashes"]:
            st.session_state["pasted_resume_images"].append({
                "name": f"pasted_screenshot_{datetime.now().strftime('%H%M%S_%f')}.png",
                "data": pasted_bytes,
                "hash": pasted_hash,
            })
            st.session_state["pasted_resume_hashes"].add(pasted_hash)

    uploaded_files = st.file_uploader(
        "＋ Attach screenshot files",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="Select multiple screenshots at once, or paste screenshots above.",
        key="resume_screenshot_uploader",
    )

    pasted_items = st.session_state["pasted_resume_images"]
    pasted_inputs = [
        ImageInput(item["name"], st.session_state["cropped_resume_images"].get(item["hash"], item["data"]))
        for item in pasted_items
    ]
    cropped_uploads = st.session_state["cropped_uploaded_images"]
    upload_inputs = [
        ImageInput(file.name, cropped_uploads[file.name]) if file.name in cropped_uploads else file
        for file in (uploaded_files or [])
    ]
    uploads = pasted_inputs + upload_inputs

    if uploads:
        st.markdown(f"**{len(uploads)} attachment(s)**")
        preview_cols = st.columns(min(4, len(uploads)))
        for i, file in enumerate(uploads):
            try:
                preview_cols[i % len(preview_cols)].image(
                    file.getvalue(), caption=f"{i + 1}. {file.name}", use_container_width=True
                )
            except Exception:
                pass

        with st.expander("✂️ Crop a screenshot before extraction", expanded=False):
            selected_index = st.selectbox(
                "Choose screenshot to crop",
                options=list(range(len(uploads))),
                format_func=lambda i: f"{i + 1}. {uploads[i].name}",
                key="crop_selected_index",
            )
            selected_file = uploads[selected_index]
            crop_image = Image.open(BytesIO(selected_file.getvalue())).convert("RGB")
            crop_w, crop_h = crop_image.size
            st.image(crop_image, caption="Selected screenshot", use_container_width=True)
            st.caption(f"Image size: {crop_w} × {crop_h}. Set the crop area as percentages.")
            c1, c2 = st.columns(2)
            left_pct = c1.slider("Left (%)", 0, 90, 0, 1, key=f"crop_left_{selected_index}")
            top_pct = c2.slider("Top (%)", 0, 90, 0, 1, key=f"crop_top_{selected_index}")
            c3, c4 = st.columns(2)
            width_pct = c3.slider("Width (%)", 10, max(10, 100-left_pct), max(10, 100-left_pct), 1, key=f"crop_width_{selected_index}")
            height_pct = c4.slider("Height (%)", 10, max(10, 100-top_pct), max(10, 100-top_pct), 1, key=f"crop_height_{selected_index}")
            x1, y1 = int(crop_w * left_pct / 100), int(crop_h * top_pct / 100)
            x2 = min(crop_w, int(crop_w * (left_pct + width_pct) / 100))
            y2 = min(crop_h, int(crop_h * (top_pct + height_pct) / 100))
            crop_preview = crop_image.crop((x1, y1, max(x1+1, x2), max(y1+1, y2)))
            st.image(crop_preview, caption="Crop preview", use_container_width=True)
            if st.button("Apply crop to this screenshot", key=f"apply_crop_{selected_index}"):
                out = BytesIO()
                crop_preview.save(out, format="PNG")
                if selected_index < len(pasted_items):
                    item = pasted_items[selected_index]
                    st.session_state["cropped_resume_images"][item["hash"]] = out.getvalue()
                else:
                    file = (uploaded_files or [])[selected_index - len(pasted_items)]
                    st.session_state["cropped_uploaded_images"][file.name] = out.getvalue()
                st.success("Crop applied.")
                st.rerun()

        if st.button("Clear all attachments", use_container_width=True):
            st.session_state["pasted_resume_images"] = []
            st.session_state["pasted_resume_hashes"] = set()
            st.session_state["cropped_resume_images"] = {}
            st.session_state["cropped_uploaded_images"] = {}
            st.session_state.pop("resume_screenshot_uploader", None)
            st.rerun()

    if len(uploads) > MAX_UPLOADS:
        st.error(f"You have added {len(uploads)} screenshots. The limit is {MAX_UPLOADS} per processing run.")
        uploads = []
    elif uploads:
        st.caption(f"{len(uploads)} screenshot(s) ready to process.")

    user_instruction = st.text_area(
        "Ask or instruct (optional)",
        placeholder="Example: Extract education and work experience. You can also type a question for the future NLQ feature.",
        height=90,
        key="resume_user_instruction",
    )
    st.caption("Local OCR extracts text from the screenshots. This instruction is captured for review; it does not yet run natural-language database queries.")

    if st.button("Extract details", type="primary", disabled=not uploads, use_container_width=True):
        with st.spinner(f"Extracting text from {len(uploads)} screenshot(s)… First run can take longer while OCR models load."):
            raw_text, warnings = extract_text(uploads)
        st.session_state["last_raw_text"] = raw_text
        st.session_state["last_warnings"] = warnings
        st.session_state["draft_fields"] = parse_fields(raw_text)
        st.session_state["draft_source_count"] = len(uploads)
        st.session_state["draft_instruction"] = user_instruction.strip()
        st.session_state["draft_saved_id"] = None
        if not raw_text.strip():
            st.error("No text was extracted. Try clearer screenshots or a different image.")
        else:
            st.success("Text extraction finished. Review the fields below before saving.")
    if st.session_state.get("last_warnings"):
        with st.expander("OCR warnings"):
            for warning in st.session_state["last_warnings"]:
                st.warning(warning)

    if st.session_state.get("draft_fields") is not None:
        fields = st.session_state["draft_fields"]
        st.subheader("Review and edit candidate details")
        st.caption("Fields not clearly present in the screenshots are left blank. Please verify extracted information before saving.")
        if st.session_state.get("draft_instruction"):
            st.info("Your instruction: " + st.session_state["draft_instruction"])
        with st.form("edit_candidate_form"):
            edited = {}
            groups = {
                "Personal Details": ["first_name","middle_name","last_name","date_of_birth","gender","nationality","marital_status","passport","hobbies","language_known"],
                "Communication Details": ["address","landmark","city","state","pincode","mobile","email"],
                "Qualification Details": ["ssc_result","ssc_board","ssc_pass_year","hsc_result","hsc_board","hsc_pass_year","graduation_degree","graduation_result","graduation_university","graduation_year","post_graduation_degree","post_graduation_result","post_graduation_university","post_graduation_year","higher_level_education"],
                "Employment & Profile": ["objectives","summary","employment_details","key_skills"],
            }
            for group, keys in groups.items():
                st.markdown(f"#### {group}")
                cols = st.columns(2)
                for i, key in enumerate(keys):
                    label = LABELS[key]
                    if key in ("objectives","summary","employment_details","key_skills","address","hobbies","language_known"):
                        edited[key] = st.text_area(label, value=fields.get(key, ""), key=f"draft_{key}", height=100 if key in ("employment_details","summary","objectives") else 70)
                    else:
                        edited[key] = cols[i % 2].text_input(label, value=fields.get(key, ""), key=f"draft_{key}")
            submitted = st.form_submit_button("Save to today's history", type="primary", use_container_width=True)
        if submitted:
            record_id = save_record(conn, edited, st.session_state.get("last_raw_text", ""), st.session_state.get("draft_source_count", 0))
            st.session_state["draft_saved_id"] = record_id
            st.success("Candidate saved to today's history.")
            st.rerun()
        with st.expander("View OCR text for troubleshooting"):
            st.text(st.session_state.get("last_raw_text", ""))

with tab_history:
    st.subheader("Search today's processed candidates")
    today_rows = conn.execute(
        "SELECT id, candidate_name, processed_at, source_count FROM history WHERE substr(processed_at,1,10)=? ORDER BY id DESC",
        (date.today().isoformat(),)
    ).fetchall()
    search = st.text_input("Search candidate name", placeholder="e.g. Sanket Kade")
    filtered = [dict(r) for r in today_rows if search.lower() in r["candidate_name"].lower()]
    st.caption(f"{len(filtered)} record(s) shown for today. Older dates are automatically cleared.")
    if not filtered:
        st.info("No matching history yet. Process a resume and save it to see it here.")
    else:
        for row in filtered:
            dt = row["processed_at"].replace("T", " ")
            with st.expander(f"{row['candidate_name']}  ·  {dt}  ·  {row['source_count']} screenshot(s)"):
                record = get_record(conn, row["id"])
                st.write(f"Record ID: {row['id']}")
                with st.form(f"history_form_{row['id']}"):
                    saved_fields = record["fields"]
                    changed = {}
                    for group, keys in {
                        "Personal Details": ["first_name","middle_name","last_name","date_of_birth","gender","nationality","marital_status","passport","hobbies","language_known"],
                        "Communication Details": ["address","landmark","city","state","pincode","mobile","email"],
                        "Qualification Details": ["ssc_result","ssc_board","ssc_pass_year","hsc_result","hsc_board","hsc_pass_year","graduation_degree","graduation_result","graduation_university","graduation_year","post_graduation_degree","post_graduation_result","post_graduation_university","post_graduation_year","higher_level_education"],
                        "Employment & Profile": ["objectives","summary","employment_details","key_skills"],
                    }.items():
                        st.markdown(f"**{group}**")
                        cols = st.columns(2)
                        for i, key in enumerate(keys):
                            if key in ("objectives","summary","employment_details","key_skills","address","hobbies","language_known"):
                                changed[key] = st.text_area(LABELS[key], value=saved_fields.get(key, ""), key=f"hist_{row['id']}_{key}", height=85)
                            else:
                                changed[key] = cols[i % 2].text_input(LABELS[key], value=saved_fields.get(key, ""), key=f"hist_{row['id']}_{key}")
                    if st.form_submit_button("Save edits"):
                        update_record(conn, row["id"], changed)
                        st.success("Changes saved.")
                        st.rerun()
                with st.expander("Original OCR text"):
                    st.text(record["raw_text"])
