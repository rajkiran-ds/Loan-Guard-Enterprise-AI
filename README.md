# 🏦 LoanGuard Enterprise AI

**An Intelligent Personal Loan Origination System using RAG, LLMs, OCR, APIs, Machine Learning, and Explainable AI.**

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B) ![XGBoost](https://img.shields.io/badge/XGBoost-2.1-brightgreen) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

LoanGuard Enterprise AI is not a chatbot — it's a full personal-loan
underwriting pipeline. A customer talks to an AI assistant grounded in the
bank's actual policy document, gets interviewed conversationally instead of
filling out a form, uploads ID documents for OCR verification, has their
CIBIL score pulled through an API abstraction, gets scored by a trained
ML model, and receives a decision from a deterministic, auditable business
rule engine — with every step logged to a database and exportable as a
professional PDF report.

## Business Problem

A bank wants to automate personal loan underwriting **without** losing
transparency, explainability, or compliance. LoanGuard solves that by
separating three concerns that are usually tangled together:

1. **Knowledge** (what does policy say?) → RAG over `bank_policy.pdf`
2. **Risk signal** (how likely is default?) → XGBoost, probability only
3. **Decision** (approve / reject / review?) → deterministic rule engine

The ML model never approves a loan. It only ever returns a number.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and
the reasoning behind the ML/policy separation.

```
Customer → RAG Assistant → OCR Verification → Mock CIBIL API →
ML Risk Model → Business Rule Engine → Loan Amount + Rate →
Database → Executive PDF Report
```

## Features

| # | Feature | Module |
|---|---|---|
| 1 | RAG Knowledge Assistant (grounded, never hallucinates) | `src/rag/` |
| 2 | Conversational AI customer interview | `pages/2_Apply_for_Loan.py` |
| 3 | OCR document verification (PAN, Aadhaar, salary slip) | `src/ocr/` |
| 4 | Mock CIBIL bureau API (swappable for a real bureau) | `src/api/` |
| 5 | XGBoost default-risk model | `src/ml/` |
| 6 | Deterministic, explainable business rule engine | `src/decision/rule_engine.py` |
| 7 | Executive KPI dashboard (Plotly) | `pages/4_Risk_Dashboard.py` |
| 8 | Full application persistence | `src/database/` |
| 9 | Executive PDF approval report | `src/decision/report_generator.py` |

## Tech Stack

- **Frontend:** Streamlit
- **LLM:** Groq API (Llama 3)
- **RAG:** LangChain, FAISS, HuggingFace embeddings
- **ML:** XGBoost, scikit-learn
- **OCR:** PaddleOCR (degrades gracefully to manual entry if unavailable)
- **Database:** SQLite via SQLAlchemy
- **Visualization:** Plotly
- **Reporting:** ReportLab
- **Deployment:** GitHub + Streamlit Cloud

## Folder Structure

```
loan_guard_enterprise_ai/
├── app.py                      # Streamlit entry point (Home)
├── config.py                   # single source of truth for constants
├── requirements.txt
├── .env.example
├── data/
│   ├── loan_dataset.csv        # synthetic by default — swap in the real Kaggle CSV
│   └── bank_policy.pdf         # RAG source document
├── pages/
│   ├── 1_Upload_Documents.py
│   ├── 2_Apply_for_Loan.py
│   ├── 3_AI_Assistant.py
│   ├── 4_Risk_Dashboard.py
│   └── 5_Approval_Report.py
├── src/
│   ├── rag/                    # loader, FAISS vectorstore, QA chain, policy PDF builder
│   ├── ocr/                    # PaddleOCR extractor
│   ├── api/                    # mock CIBIL bureau client + optional FastAPI wrapper
│   ├── ml/                     # dataset generator, training script, inference wrapper
│   ├── decision/                # rule engine, explainer, PDF report generator
│   ├── database/               # SQLAlchemy models + repository layer
│   └── utils/                  # logger, validators, UI theme
└── docs/architecture.md
```

## ML Pipeline

Trained on [this Kaggle loan-approval dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset).
Kaggle requires authenticated download, so `src/ml/generate_sample_dataset.py`
ships a schema-compatible **synthetic** dataset (4,000 rows) so the pipeline
runs out of the box. To use the real dataset:

```bash
kaggle datasets download -d architsharma01/loan-approval-prediction-dataset
unzip loan-approval-prediction-dataset.zip -d data/
# rename columns to match config.ML_FEATURE_COLUMNS + config.ML_TARGET_COLUMN
python -m src.ml.train_model
```

Current synthetic-data metrics (`src/ml/artifacts/metrics.json`):

| ROC AUC | Precision | Recall | F1 |
|---|---|---|---|
| 0.92 | 0.78 | 0.77 | 0.78 |

## RAG Pipeline

1. `src/rag/build_policy_pdf.py` — generates `data/bank_policy.pdf` (replace with your own anytime)
2. `src/rag/loader.py` — loads + chunks the PDF (`RecursiveCharacterTextSplitter`)
3. `src/rag/vectorstore.py` — embeds chunks with a HuggingFace sentence-transformer, indexes in FAISS, caches to disk
4. `src/rag/qa_chain.py` — retrieves top-k chunks, forces Groq/Llama 3 to answer **only** from that context, returns a fixed refusal message otherwise

## New Feature: Multilingual Voice AI

The **Apply for Loan** and **AI Assistant** pages are voice-first and
multilingual. A customer can complete the entire interview and ask policy
questions by speaking or typing in **English, Telugu, Hindi, Tamil, or
Kannada** — the AI detects the language automatically and always replies
in that same language, with the answer also read aloud.

The bank policy document itself is **not translated**. `bank_policy.pdf`
stays English-only in FAISS; only the answer-generation step changes,
instructing the LLM to reason over the retrieved English chunks but
respond in the customer's language. Retrieval (LangChain + FAISS) is
completely unchanged. See `src/rag/multilingual_chat.py`.

New modules (nothing existing was rewritten):

| Module | Purpose |
|---|---|
| `src/utils/language.py` | Detects the customer's language → ISO 639-1 code (`en`/`te`/`hi`/`ta`/`kn`). Unicode script check first, `langdetect` second, LLM fallback last — never raises, defaults to `en`. |
| `src/voice/stt.py` | Speech-to-text via faster-whisper (falls back to openai-whisper). Degrades to `engine_available=False` if neither is installed, so the UI falls back to typed text. |
| `src/voice/tts.py` | Text-to-speech via gTTS. Returns `None` on failure (offline, unsupported language) instead of raising — the calling page just skips the audio player. |
| `src/rag/multilingual_chat.py` | Wraps the existing `PolicyAssistant` (unmodified) with a language-aware prompt, plus a `translate_text()` helper used to render the final loan decision in the customer's language. |

## Voice Workflow

1. **Apply for Loan** — toggle "🎙️ Speak my answers"; each interview field
   has its own mic input. The transcribed text is shown for confirmation,
   pre-fills the form field below (spoken answers take priority over OCR
   pre-fill, both are just defaults the customer can still edit), and the
   detected language becomes the interview's active language.
2. **"🔊 Hear questions"** button reads the interview questions aloud in
   the active language.
3. On submission, the ML pipeline, business rule engine, and database
   record are **completely unchanged** — voice input only ever produces
   the same normalized text/number values a typed answer would.
4. If the interview's active language wasn't English, the approval
   explanation is additionally translated and read aloud via TTS
   (English explanation and the DB record stay as-is).
5. **AI Assistant** — toggle "🎙️ Voice" to ask a policy question by mic;
   language is auto-detected per message (or pinned via the sidebar
   language selector), and every answer includes an audio player.

## Supported Languages

| Language | ISO Code |
|---|---|
| English | `en` |
| Telugu | `te` |
| Hindi | `hi` |
| Tamil | `ta` |
| Kannada | `kn` |

## Installation of Whisper

Voice input needs one Whisper backend — `faster-whisper` (recommended,
CPU-friendly CTranslate2 build) is tried first, `openai-whisper` second.
Both are in `requirements.txt`; installing either is enough:

```bash
pip install faster-whisper      # recommended — faster on CPU
# or
pip install openai-whisper      # requires ffmpeg on PATH
```

If neither is installed (or model download fails), voice input is
disabled automatically and the interview/assistant fall back to typed
text — nothing else breaks.

## Microphone Permissions

Voice input uses Streamlit's native `st.audio_input`, which records
through the browser's microphone API:

- **Browser:** grant microphone access when prompted (usually a padlock/
  camera icon in the address bar). If denied, re-enable it in the site's
  browser permissions and reload the page.
- **Streamlit Cloud / remote deployments:** the browser requires HTTPS
  (or `localhost`) to allow microphone access — plain HTTP on a remote
  host will silently block the mic prompt.
- If the mic is unavailable or blocked, every voice-enabled page still
  works fully in typed-text mode.

## OCR Workflow

Upload PAN / Aadhaar / salary slip images on the **Upload Documents** page →
PaddleOCR extracts raw text → regex parsers pull out PAN, Aadhaar, and
income → fields pre-fill the **Apply for Loan** interview, editable by hand.

## API Integration

`src/api/cibil_api.py` exposes `CibilBureauClient.fetch_score(pan, dob)`.
Today it's backed by a deterministic mock (`MockCibilProvider`); swapping to
a real bureau means writing one new class satisfying the same
`BureauProvider` interface. A thin FastAPI wrapper is included so the same
logic can also be served as a literal `POST /cibil` HTTP endpoint:

```bash
uvicorn src.api.cibil_api:api_app --reload --port 8000
```

## Business Rule Engine

Four rules, evaluated in priority order, fully documented in
`config.LENDING_RULES` and `src/decision/rule_engine.py`:

| Rule | CIBIL | Risk | Max DTI | Outcome | Max Loan | Rate |
|---|---|---|---|---|---|---|
| 1 — Prime | ≥ 780 | Low | < 30% | Approved | 10× salary | 9.5% |
| 2 — Standard | 720–779 | Low/Medium | < 40% | Approved | 8× salary | 11% |
| 3 — Conditional | 650–719 | Medium, 2y+ employment | — | Approved with Review | 5× salary | 13.5% |
| 4 — Reject | < 650, or High risk, or > 3 late payments | — | — | Rejected | — | — |

Interest rate = 9% base + risk adjustment (0/2/4%) + 1% if DTI > 40% +
1.5% if CIBIL < 700, clamped to **9%–18%**.

## Installation

```bash
git clone https://github.com/rajkiran-ds/loan-guard-enterprise-ai.git
cd loan_guard_enterprise_ai
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env      # then add your GROQ_API_KEY

python -m src.rag.build_policy_pdf       # generate the policy PDF (or supply your own)
python -m src.ml.generate_sample_dataset # generate the synthetic dataset (or supply the Kaggle CSV)
python -m src.ml.train_model             # train the XGBoost risk model

streamlit run app.py
```

## Environment Variables

See [`.env.example`](.env.example) — at minimum you need `GROQ_API_KEY`
for the AI Assistant page; every other feature works without any external
API key.

## Streamlit Deployment

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at `app.py`.
3. Add `GROQ_API_KEY` under **App settings → Secrets** as:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
4. PaddleOCR's model weights download on first use — if Streamlit Cloud's
   environment blocks that, the OCR page degrades gracefully to manual entry.

## GitHub Deployment

```bash
git init
git add .
git commit -m "LoanGuard Enterprise AI — initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/loan-guard-enterprise-ai.git
git push -u origin main
```

## Future Improvements

- Real bureau integration (Experian/Equifax/CIBIL live API) via a new `BureauProvider`
- Model monitoring + drift detection on the XGBoost risk model
- Role-based auth for underwriters reviewing "Approved with Review" cases
- Multi-document RAG (add loan agreement templates, FAQ docs)
- Async document verification queue for high application volume

## Screenshots

_Add screenshots of Home, Apply for Loan, AI Assistant, Risk Dashboard, and
Approval Report here before publishing._

## License

MIT
