🏦 LoanGuard Enterprise AI
An Intelligent Personal Loan Origination System using RAG, LLMs, OCR, APIs, Machine Learning, and Agentic AI.

https://img.shields.io/badge/Python-3.10+-blue https://img.shields.io/badge/Streamlit-1.38-FF4B4B https://img.shields.io/badge/PyTorch-2.2-EE4C2C https://img.shields.io/badge/LangGraph-0.0.30-orange https://img.shields.io/badge/Docker-Enabled-2496ED

Overview
LoanGuard Enterprise AI is not a chatbot — it's a full personal-loan
underwriting pipeline. A customer talks to an AI assistant grounded in the
bank's actual policy document, gets interviewed conversationally instead of
filling out a form, uploads ID documents for OCR verification, has their
CIBIL score pulled through an API abstraction, gets scored by a trained
PyTorch ML model, and is routed through an agentic workflow that includes
a deterministic, auditable business rule engine and a human-in-the-loop
review step for borderline cases.

Business Problem
A bank wants to automate personal loan underwriting without losing
transparency, explainability, or compliance. LoanGuard solves that by
separating three concerns that are usually tangled together:

Knowledge (what does policy say?) → RAG over bank_policy.pdf

Risk signal (how likely is default?) → PyTorch MLP, probability only

Decision (approve / reject / review?) → LangGraph agent orchestrating a deterministic rule engine

The ML model never approves a loan. It only ever returns a number.

Architecture
See docs/architecture.md for the full diagram and
the reasoning behind the ML/policy separation.

text
Customer → RAG Assistant → OCR Verification → Mock CIBIL API →
PyTorch Risk Model → LangGraph Agent (Rule Engine → Human-in-the-Loop) → 
Loan Amount + Rate → Database → Executive PDF Report
Features
#	Feature	Module
1	RAG Knowledge Assistant (grounded, never hallucinates)	src/rag/
2	Conversational AI customer interview	pages/2_Apply_for_Loan.py
3	OCR document verification (PAN, Aadhaar, salary slip)	src/ocr/
4	Mock CIBIL bureau API (swappable for a real bureau)	src/api/
5	PyTorch MLP default-risk model (selected via 6-model experimentation)	src/ml/
6	Agentic Workflow with Human-in-the-Loop (LangGraph)	src/agent/
7	Deterministic, explainable business rule engine	src/decision/rule_engine.py
8	Executive KPI dashboard (Plotly)	pages/4_Risk_Dashboard.py
9	Full application persistence	src/database/
10	Executive PDF approval report	src/decision/report_generator.py
11	Containerized deployment	Dockerfile
Tech Stack
Frontend: Streamlit

LLM: Groq API (Llama 3)

Agentic AI: LangGraph, LangChain

RAG: FAISS, HuggingFace embeddings

ML: PyTorch, TensorFlow, scikit-learn

OCR: PaddleOCR (degrades gracefully to manual entry if unavailable)

Database: SQLite via SQLAlchemy

Visualization: Plotly

Reporting: ReportLab

Deployment: Docker, Docker Compose, GitHub + Streamlit Cloud

Folder Structure
text
loan_guard_enterprise_ai/
├── app.py                      # Streamlit entry point (Home)
├── config.py                   # single source of truth for constants
├── requirements.txt
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Container orchestration
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
│   ├── agent/                  # LangGraph state machine for underwriting orchestration
│   ├── ml/                     # dataset generator, experimentation, training, inference
│   ├── decision/                # rule engine, explainer, PDF report generator
│   ├── database/               # SQLAlchemy models + repository layer
│   └── utils/                  # logger, validators, UI theme
└── docs/architecture.md
ML Pipeline & Experimentation
Trained on this Kaggle loan-approval dataset.
Kaggle requires authenticated download, so src/ml/generate_sample_dataset.py
ships a schema-compatible synthetic dataset (4,000 rows) so the pipeline
runs out of the box. To use the real dataset:

bash
kaggle datasets download -d architsharma01/loan-approval-prediction-dataset
unzip loan-approval-prediction-dataset.zip -d data/
# rename columns to match config.ML_FEATURE_COLUMNS + config.ML_TARGET_COLUMN
Model Experimentation:
src/ml/experimentation.py performs stratified 5-fold cross-validation across six algorithms: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, a PyTorch MLP, and a TensorFlow MLP.

Based on evaluation metrics (optimizing for recall on the approved class to minimize false positives in lending), the PyTorch MLP was selected for production.

Production Metrics (src/ml/artifacts/metrics.json):

Model	ROC AUC	Precision	Recall	F1
PyTorch MLP (Production)	0.9375	0.7838	0.7961	0.7899
XGBoost (Baseline)	0.9297	0.7918	0.7439	0.7667
Agentic Workflow (LangGraph)
The underwriting decision process is orchestrated by a LangGraph StateGraph in src/agent/graph.py. This separates ML inference from deterministic rules while allowing for human intervention:

score_ml: The PyTorch MLP evaluates the application and outputs a default probability and risk tier.

evaluate_rules: The deterministic rule engine evaluates the applicant against policy rules (Prime, Standard, Conditional, Reject).

Conditional Edge: If the application triggers a manual review rule (e.g., Rule 3 — Conditional), the graph routes to human_review.

human_review: The graph uses interrupt() to pause execution and persist state via MemorySaver. The application is flagged for an underwriter.

finalize: Once the human approves/rejects (or if no review is needed), the graph resumes and finalizes the decision.

RAG Pipeline
src/rag/build_policy_pdf.py — generates data/bank_policy.pdf (replace with your own anytime)

src/rag/loader.py — loads + chunks the PDF (RecursiveCharacterTextSplitter)

src/rag/vectorstore.py — embeds chunks with a HuggingFace sentence-transformer, indexes in FAISS, caches to disk

src/rag/qa_chain.py — retrieves top-k chunks, forces Groq/Llama 3 to answer only from that context, returns a fixed refusal message otherwise

Multilingual Voice AI
The Apply for Loan and AI Assistant pages are voice-first and
multilingual. A customer can complete the entire interview and ask policy
questions by speaking or typing in English, Telugu, Hindi, Tamil, or
Kannada — the AI detects the language automatically and always replies
in that same language, with the answer also read aloud.

The bank policy document itself is not translated. bank_policy.pdf
stays English-only in FAISS; only the answer-generation step changes,
instructing the LLM to reason over the retrieved English chunks but
respond in the customer's language. Retrieval (LangChain + FAISS) is
completely unchanged. See src/rag/multilingual_chat.py.

Module	Purpose
src/utils/language.py	Detects the customer's language → ISO 639-1 code (en/te/hi/ta/kn). Unicode script check first, langdetect second, LLM fallback last — never raises, defaults to en.
src/voice/stt.py	Speech-to-text via faster-whisper (falls back to openai-whisper). Degrades to engine_available=False if neither is installed, so the UI falls back to typed text.
src/voice/tts.py	Text-to-speech via gTTS. Returns None on failure (offline, unsupported language) instead of raising — the calling page just skips the audio player.
src/rag/multilingual_chat.py	Wraps the existing PolicyAssistant (unmodified) with a language-aware prompt, plus a translate_text() helper used to render the final loan decision in the customer's language.
Voice Workflow
Apply for Loan — toggle "🎙️ Speak my answers"; each interview field
has its own mic input. The transcribed text is shown for confirmation,
pre-fills the form field below (spoken answers take priority over OCR
pre-fill, both are just defaults the customer can still edit), and the
detected language becomes the interview's active language.

"🔊 Hear questions" button reads the interview questions aloud in
the active language.

On submission, the ML pipeline, business rule engine, and database
record are completely unchanged — voice input only ever produces
the same normalized text/number values a typed answer would.

If the interview's active language wasn't English, the approval
explanation is additionally translated and read aloud via TTS
(English explanation and the DB record stay as-is).

AI Assistant — toggle "🎙️ Voice" to ask a policy question by mic;
language is auto-detected per message (or pinned via the sidebar
language selector), and every answer includes an audio player.

Supported Languages
Language	ISO Code
English	en
Telugu	te
Hindi	hi
Tamil	ta
Kannada	kn
Installation of Whisper
Voice input needs one Whisper backend — faster-whisper (recommended,
CPU-friendly CTranslate2 build) is tried first, openai-whisper second.
Both are in requirements.txt; installing either is enough:

bash
pip install faster-whisper      # recommended — faster on CPU
# or
pip install openai-whisper      # requires ffmpeg on PATH
If neither is installed (or model download fails), voice input is
disabled automatically and the interview/assistant fall back to typed
text — nothing else breaks.

Microphone Permissions
Voice input uses Streamlit's native st.audio_input, which records
through the browser's microphone API:

Browser: grant microphone access when prompted (usually a padlock/
camera icon in the address bar). If denied, re-enable it in the site's
browser permissions and reload the page.

Streamlit Cloud / remote deployments: the browser requires HTTPS
(or localhost) to allow microphone access — plain HTTP on a remote
host will silently block the mic prompt.

If the mic is unavailable or blocked, every voice-enabled page still
works fully in typed-text mode.

OCR Workflow
Upload PAN / Aadhaar / salary slip images on the Upload Documents page →
PaddleOCR extracts raw text → regex parsers pull out PAN, Aadhaar, and
income → fields pre-fill the Apply for Loan interview, editable by hand.

API Integration
src/api/cibil_api.py exposes CibilBureauClient.fetch_score(pan, dob).
Today it's backed by a deterministic mock (MockCibilProvider); swapping to
a real bureau means writing one new class satisfying the same
BureauProvider interface. A thin FastAPI wrapper is included so the same
logic can also be served as a literal POST /cibil HTTP endpoint:

bash
uvicorn src.api.cibil_api:api_app --reload --port 8000
Business Rule Engine
Four rules, evaluated in priority order, fully documented in
config.LENDING_RULES and src/decision/rule_engine.py:

Rule	CIBIL	Risk	Max DTI	Outcome	Max Loan	Rate
1 — Prime	≥ 780	Low	< 30%	Approved	10× salary	9.5%
2 — Standard	720–779	Low/Medium	< 40%	Approved	8× salary	11%
3 — Conditional	650–719	Medium, 2y+ employment	—	Approved with Review	5× salary	13.5%
4 — Reject	< 650, or High risk, or > 3 late payments	—	—	Rejected	—	—
Interest rate = 9% base + risk adjustment (0/2/4%) + 1% if DTI > 40% +
1.5% if CIBIL < 700, clamped to 9%–18%.

Installation (Local)
bash
git clone https://github.com/rajkiran-ds/loan-guard-enterprise-ai.git
cd loan_guard_enterprise_ai
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env      # then add your GROQ_API_KEY

python -m src.rag.build_policy_pdf       # generate the policy PDF (or supply your own)
python -m src.ml.generate_sample_dataset # generate the synthetic dataset (or supply the Kaggle CSV)
python -m src.ml.train_model             # train and save the PyTorch MLP risk model

streamlit run app.py
Docker Deployment
The entire application is containerized for reliable deployment across environments.

bash
docker compose up --build -d
The app will be accessible at http://localhost:8501. The docker-compose.yml mounts the data/ directory to persist your SQLite database and FAISS indexes across container restarts.

Environment Variables
See .env.example — at minimum you need GROQ_API_KEY
for the AI Assistant page; every other feature works without any external
API key.

Streamlit Deployment
Push this repo to GitHub.

On share.streamlit.io, point a new app at app.py.

Add GROQ_API_KEY under App settings → Secrets as:

toml
GROQ_API_KEY = "your_key_here"
PaddleOCR's model weights download on first use — if Streamlit Cloud's
environment blocks that, the OCR page degrades gracefully to manual entry.

Future Improvements
Real bureau integration (Experian/Equifax/CIBIL live API) via a new BureauProvider

Model monitoring + drift detection on the PyTorch MLP risk model

Role-based auth for underwriters reviewing "Approved with Review" cases

Multi-document RAG (add loan agreement templates, FAQ docs)

Async document verification queue for high application volume

Author: Pendyala Raj kiran Reddy
LinkedIn · GitHub

did you added the limitations of the project?
Analyze the Request:

User asks: "did you added the limitations of the project?"

Context: The user previously provided the original README.md and asked me to update it with the new features (PyTorch MLP, LangGraph, Docker) without explicitly stating "we updated this" and without hallucinating, following the original flow.

Reviewing the provided original README: It did not have a "Limitations" section. It went from "Future Improvements" to "Author".

Reviewing my updated README: I kept the exact structure, replacing/augmenting sections. I did not add a "Limitations" section because it wasn't in the original, and the prompt said "Don't add any hallucinated information and follow the flow of our existing readme".

Evaluate the need for Limitations:

Adding limitations is actually a great idea for a portfolio project, especially for an AI/ML role. It shows self-awareness, engineering maturity, and honesty.

If the user wants limitations, I should provide a well-crafted, professional section highlighting the current boundaries of the project (e.g., synthetic data, mock CIBIL API, no real auth for underwriters yet, etc.) and explain why it's good to include it.

I should offer to add it back in.

Identify actual limitations based on the project:

Data: Using a synthetic dataset (or a small Kaggle one) rather than real, large-scale banking data.

CIBIL: It's a mock API, not a real bureau integration.

Auth: No real user authentication/authorization for customers vs. underwriters (it's a single-user Streamlit app for demo purposes).

Scale: SQLite + local FAISS + Streamlit is a monolith prototype, not a distributed production system.

OCR: PaddleOCR degrades to manual entry if not available; real-world OCR of blurry bank slips is much harder.

Agentic Workflow: Human-in-the-loop is simulated via LangGraph's memory saver in a local process, not a real ticketing/queueing system for underwriters.

Formulate the response:

Acknowledge that I did not add it because the original README didn't have one, and I followed the "don't hallucinate / follow the flow" instruction strictly.

Explain why adding a limitations section is actually highly recommended for interviews.

Provide the exact Markdown block for a Limitations section, tailored to the current stack.

Show exactly where to insert it in the README (usually before "Future Improvements").

Provide the interview talking points for these limitations (e.g., turning a weakness into a strength).

You caught me! I did not add a limitations section.

My strict instruction was to "follow the flow of our existing readme," and your original README went straight from Streamlit Deployment to Future Improvements without a limitations section. I didn't want to hallucinate a new section and break the flow you already had.

However, as a Senior AI Architect, I highly recommend adding one. Including project limitations is a massive green flag for interviewers. It shows engineering maturity, self-awareness, and honesty. It proves you know exactly where the boundaries of your current architecture are.

Here is the exact Limitations section I would add. Insert this right above the "Future Improvements" section:

Limitations
While LoanGuard demonstrates a production-style architecture, the current implementation has the following scope boundaries:

Mock Bureau Integration: CIBIL data is fetched via a deterministic mock provider. A production system would require secure, authenticated integration with a live bureau (Experian/Equifax/CIBIL).

Synthetic Data: The default PyTorch MLP is trained on a schema-compatible synthetic dataset. While the pipeline supports the real Kaggle dataset, real-world banking data requires far more rigorous feature engineering and bias mitigation.

Authentication: The Streamlit application does not currently implement user authentication. Any user can access the customer interview or the Risk Dashboard. A production system requires RBAC (Role-Based Access Control) to separate customer access from underwriter access.

Local Queue for Human Review: The LangGraph human-in-the-loop workflow uses an in-memory MemorySaver checkpointer. If the Streamlit server restarts, pending reviews are lost. A production system would use a persistent database checkpointer (e.g., Postgres) and a proper ticketing queue.

Monolithic Deployment: The app is deployed as a single container. At enterprise scale, the OCR extraction, RAG pipeline, and ML inference would be decoupled into microservices to scale independently.

🎤 Why this is powerful in the interview:
If an interviewer asks, "What are the weaknesses of this project?" or "What would you change before putting this in production?"—you now have a perfect, pre-planned script.

You can say:

"I designed it as a monolithic proof-of-concept to validate the agentic workflow and ML integration. However, I documented the production limitations in the README. For example, I used an in-memory checkpointer for LangGraph; in production, I would swap that for a Postgres checkpointer so underwriter review queues survive server restarts. I also didn't implement RBAC, which is critical for a real banking app."

It turns a weakness into a demonstration of system design foresight.

Paste that markdown block into your README.md and push to GitHub!

where to add these limitations?
Analyze the Request:

User wants to know exactly where to put the "Limitations" section in their README.md.

The previous response provided the content but the user wants placement clarity.

Review the Current README Structure:

Overview

Business Problem

Architecture

Features

Tech Stack

Folder Structure

ML Pipeline & Experimentation

Agentic Workflow (LangGraph)

RAG Pipeline

Multilingual Voice AI (Subsections: Voice Workflow, Supported Languages, Installation of Whisper, Microphone Permissions)

OCR Workflow

API Integration

Business Rule Engine

Installation (Local)

Docker Deployment

Environment Variables

Streamlit Deployment

Future Improvements (Current ending)

Author

Determine the Best Placement:

Logical flow: ... -> Deployment -> Limitations -> Future Improvements.

Putting it right before "Future Improvements" makes perfect sense. "Here are the things this project doesn't do right now, and here is how I plan to fix them (Future Improvements)."

I'll point to the exact section heading in the provided markdown so they can easily use Ctrl+F.

Formulate the Response:

State the exact location clearly (right above ## Future Improvements).

Provide the surrounding context so they can see exactly where it slots in.

Keep it brief and actionable.

You should place it right above the Future Improvements section at the very bottom of the README.

Here is exactly how the bottom of your README should look. You can scroll to the bottom of your file and copy/paste it so it fits perfectly between "Streamlit Deployment" and "Future Improvements".

Streamlit Deployment
Push this repo to GitHub.

On share.streamlit.io, point a new app at app.py.

Add GROQ_API_KEY under App settings → Secrets as:

toml
GROQ_API_KEY = "your_key_here"
PaddleOCR's model weights download on first use — if Streamlit Cloud's
environment blocks that, the OCR page degrades gracefully to manual entry.

Limitations
While LoanGuard demonstrates a production-style architecture, the current implementation has the following scope boundaries:

Mock Bureau Integration: CIBIL data is fetched via a deterministic mock provider. A production system would require secure, authenticated integration with a live bureau (Experian/Equifax/CIBIL).

Synthetic Data: The default PyTorch MLP is trained on a schema-compatible synthetic dataset. While the pipeline supports the real Kaggle dataset, real-world banking data requires far more rigorous feature engineering and bias mitigation.

Authentication: The Streamlit application does not currently implement user authentication. Any user can access the customer interview or the Risk Dashboard. A production system requires RBAC (Role-Based Access Control) to separate customer access from underwriter access.

Local Queue for Human Review: The LangGraph human-in-the-loop workflow uses an in-memory MemorySaver checkpointer. If the Streamlit server restarts, pending reviews are lost. A production system would use a persistent database checkpointer (e.g., Postgres) and a proper ticketing queue.

Monolithic Deployment: The app is deployed as a single container. At enterprise scale, the OCR extraction, RAG pipeline, and ML inference would be decoupled into microservices to scale independently.

Future Improvements
Real bureau integration (Experian/Equifax/CIBIL live API) via a new BureauProvider

Model monitoring + drift detection on the PyTorch MLP risk model

Role-based auth for underwriters reviewing "Approved with Review" cases

Multi-document RAG (add loan agreement templates, FAQ docs)

Async document verification queue for high application volume

Author: Pendyala Raj kiran Reddy
LinkedIn · GitHub

