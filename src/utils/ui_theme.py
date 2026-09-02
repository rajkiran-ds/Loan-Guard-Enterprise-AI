"""
src/utils/ui_theme.py
------------------------
Shared "dark enterprise banking" CSS injected on every page so the six
pages (Home, Apply, AI Assistant, Upload Documents, Risk Dashboard,
Approval Report) look like one cohesive product instead of six separate
Streamlit demos.
"""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
:root {
    --lg-navy: #0B1F3A;
    --lg-navy-light: #14294D;
    --lg-gold: #C9A648;
    --lg-green: #1B7A3D;
    --lg-red: #B00020;
    --lg-text: #E7EBF3;
    --lg-muted: #9AA7BD;
}

.stApp {
    background: radial-gradient(circle at 20% 0%, #0F2444 0%, #0A1830 55%, #060F20 100%);
    color: var(--lg-text);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081327 0%, #0B1F3A 100%);
    border-right: 1px solid rgba(201, 166, 72, 0.15);
}

h1, h2, h3 {
    color: var(--lg-text) !important;
    letter-spacing: 0.2px;
}

.lg-kicker {
    color: var(--lg-gold);
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}

.lg-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(201, 166, 72, 0.18);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
}

.lg-metric {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 0.9rem 1rem;
}

.lg-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.4px;
}
.lg-badge-approved { background: rgba(27,122,61,0.18); color: #4FD37E; border: 1px solid rgba(79,211,126,0.35); }
.lg-badge-review    { background: rgba(201,166,72,0.18); color: #E4C766; border: 1px solid rgba(228,199,102,0.35); }
.lg-badge-rejected  { background: rgba(176,0,32,0.18); color: #FF7E8F; border: 1px solid rgba(255,126,143,0.35); }

.stButton > button {
    background: linear-gradient(135deg, var(--lg-gold), #A9853A);
    color: #0B1F3A;
    font-weight: 700;
    border: none;
    border-radius: 8px;
}
.stButton > button:hover {
    filter: brightness(1.08);
}

hr { border-color: rgba(255,255,255,0.08); }
</style>
"""


def apply_theme(page_title: str) -> None:
    st.set_page_config(page_title=f"{page_title} · LoanGuard Enterprise AI", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)


def badge(text: str, kind: str) -> str:
    css_class = {"Approved": "lg-badge-approved", "Approved with Review": "lg-badge-review",
                 "Rejected": "lg-badge-rejected"}.get(kind, "lg-badge-review")
    return f'<span class="lg-badge {css_class}">{text}</span>'
