# Resume Screenshot Autofill

A local-first Streamlit app that reads resume screenshots with PaddleOCR, fills an editable candidate form, and keeps a searchable history. No paid API, no external OCR, and no screenshot or candidate data leaves your machine.

## Flow
Screenshot or PDF composer (paste / upload) → attachment queue → image validation + preprocessing → PaddleOCR → field extraction with source evidence → validation + conflict detection → editable review form → save → SQLite → searchable history → optional read-only questions.

## Project layout
| Path | Responsibility |
|---|---|
| `app.py` | Streamlit entry point (wiring only) |
| `autofill/ui/` | dark theme, composer, review form, history view |
| `autofill/attachments.py` | attachment queue: dedupe, limit, reorder, crop |
| `autofill/pdfs.py` | PDF input: embedded text read directly, scanned pages rendered for OCR |
| `autofill/structure.py` | OCR text and jobs as tables (by section, line by line, per job) |
| `autofill/imaging.py` | image validation and OCR preprocessing |
| `autofill/ocr_engine.py` | PaddleOCR wrapper, result parsing, cache, oneDNN fix |
| `autofill/extraction.py` | rule-based field extraction + evidence |
| `autofill/validation.py` | format checks, conflicts, review flags |
| `autofill/geo.py`, `autofill/data/pincodes.csv` | offline pincode → city/state (19,486 India Post pincodes) |
| `autofill/jobs.py` | splits employment text into one entry per job |
| `autofill/db/repository.py` | repository interface + SQLite implementation + migrations |
| `autofill/history.py`, `retention.py` | save/search, daily retention |
| `autofill/nlq.py` | optional read-only questions (rule-based, never generates SQL) |
| `tests/` | pytest suite |

## Setup (Windows, Python 3.11 64-bit)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m autofill.selfcheck     # verifies local OCR works (first run downloads models)
streamlit run app.py
```
Linux: `python3.11 -m venv .venv && source .venv/bin/activate`, then the same pip/streamlit commands.
Developers: `pip install -r requirements-dev.txt && pytest`.

## Troubleshooting: `ConvertPirAttribute2RuntimeAttribute not support`
This is a PaddlePaddle 3.x bug in its oneDNN (MKLDNN) CPU path. The app now disables oneDNN (`enable_mkldnn=False`, `FLAGS_use_mkldnn=0`) and retries once if it still sees the error. If `python -m autofill.selfcheck` still reports it, install the known-good build: `pip install paddlepaddle==3.0.0`.

## City / State and jobs
- City and State are filled from the **pincode** using the bundled India Post table (no network). City is the India Post district. A blank City/State is also filled from the pincode when you save. If a labelled State disagrees with the pincode, the field is flagged.
- **PDF resumes** (`RESUME_MAX_PDF_PAGES`=10 pages, `RESUME_PDF_SCALE`=2.0): a PDF with a text layer is read directly in milliseconds, no OCR. Scanned PDFs are rendered and OCR'd like screenshots.
- **Employment** shows one box per job (plus one spare box to add another). Jobs are detected from date ranges such as `Jan 2020 - Mar 2022`, from `Worked as …` / `Role:` lines (company line above), or from lone durations such as `1.5 Year`; a Company / Role / Duration / Details table summarises them; check the split, then save. Boxes are stored in the same `employment_details` field, separated by `---`, so old records still open.

## Behaviour
- Up to 30 screenshots per candidate run, from paste (clipboard button) and/or multi-file upload/drag-and-drop. Thumbnails can be removed, reordered and cropped before extraction.
- Nothing is invented. Unreadable fields stay blank ("Not found — left blank"). A ⚠️ marks fields to verify: invalid format, several conflicting values, low OCR confidence, or a value guessed from layout. Hover the field's ? for the reason and the source screenshot/text.
- Saving never blocks on warnings, and saving the same draft twice updates the same entry. Re-processing a candidate creates a new entry.
- History keeps **today only** by default (set `RESUME_RETENTION_DAYS=N` to keep N extra days). Old records are purged at start and at most once a minute afterwards. Screenshots are never written to disk.
- The history tab is a table with search; only the selected record gets an editor, so it stays fast with ~1,000 records a day.
- Existing `resume_history.db` files are migrated in place (a one-time `.pre-migration.bak` copy is made if the file holds data).

## Configuration (environment variables)
`RESUME_DB_PATH`, `RESUME_MAX_UPLOADS` (30), `RESUME_MAX_IMAGE_MB` (20), `RESUME_OCR_MAX_EDGE` (1280), `RESUME_OCR_MIN_SCORE` (0.30), `RESUME_RETENTION_DAYS` (0), `RESUME_OCR_PROFILE` (`fast` default, or `accurate`), `RESUME_OCR_THREADS` (CPU cores), `RESUME_OCR_MKLDNN` (0; set 1 only with paddlepaddle==3.0.0).

## Speed (RapidOCR backend)
`requirements.txt` now installs **RapidOCR** (ONNX Runtime). It runs several screenshots at the same time (`RESUME_OCR_WORKERS`, default half your cores, max 4). PaddleOCR is still used as a fallback if RapidOCR is not installed, or force it with `RESUME_OCR_BACKEND=paddle`. Per-screenshot times appear under **Timing per screenshot**.

**Overlapping screenshots:** rows already read in an earlier screenshot are skipped before OCR (`autofill/overlap.py`), so scrolled captures of the same page cost roughly one page of OCR, not one per screenshot. Best of all: upload the resume as a **PDF** (text PDFs need no OCR and read in milliseconds).

## Older speed notes (PaddleOCR)
Fastest first: text PDFs skip OCR entirely. For screenshots the biggest single gain is `pip install paddlepaddle==3.0.0` and `RESUME_OCR_MKLDNN=1` (oneDNN is off by default because it crashes on 3.1+). Screenshots are now downscaled to 1280px by default (`RESUME_OCR_MAX_EDGE`).
The default `fast` profile uses PaddleOCR's mobile models without the per-line orientation pass, uses all CPU cores, and loads the models in the background when the app starts. Per-screenshot times are shown under **Timing per screenshot**; identical screenshots are cached. For `accurate` mode set `RESUME_OCR_PROFILE=accurate` (PowerShell: `$env:RESUME_OCR_PROFILE="accurate"` before `streamlit run app.py`).

## PostgreSQL later
The app only talks to `HistoryRepository` (`autofill/db/repository.py`). Add a `PostgresHistoryRepository` with the same methods and swap it in `app.py:get_repo()`. Not implemented yet.

## Accuracy
This is a conservative rule-based extractor, not a guarantee. It is tested on a synthetic resume, not on your real screenshots. Measure it on a labelled set of your own screenshots before relying on it, and always review the form before saving.

## Cloud notes
Free hosts may not have the memory/build limits PaddlePaddle needs, and ephemeral filesystems lose history on restart (use `RESUME_DB_PATH` on a persistent volume). Never commit the database or candidate data.
