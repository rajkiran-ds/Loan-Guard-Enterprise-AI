"""
pages/4_Risk_Dashboard.py
-----------------------------
FEATURE 7 — Executive Dashboard. KPIs + five Plotly charts over every
application saved to the database so far.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.database.db import list_applications_df
from src.utils.ui_theme import apply_theme

apply_theme("Risk Dashboard")

st.markdown('<div class="lg-kicker">Feature 7</div>', unsafe_allow_html=True)
st.title("📊 Executive Risk Dashboard")

df = list_applications_df()

if df.empty:
    st.info("No applications yet. Submit one via **Apply for Loan** to populate this dashboard.")
    st.stop()

decided = df[df["decision"].notna()]
total = len(df)
approved = decided[decided["decision"].isin(["Approved", "Approved with Review"])]
approval_rate = len(approved) / len(decided) if len(decided) else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(f'<div class="lg-metric"><div class="lg-kicker">Total Applications</div><h2>{total}</h2></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="lg-metric"><div class="lg-kicker">Approval Rate</div><h2>{approval_rate:.0%}</h2></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="lg-metric"><div class="lg-kicker">Avg CIBIL</div><h2>{df["cibil_score"].mean():.0f}</h2></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="lg-metric"><div class="lg-kicker">Avg Loan</div><h2>₹{df["approved_loan_amount"].mean():,.0f}</h2></div>', unsafe_allow_html=True)
k5.markdown(f'<div class="lg-metric"><div class="lg-kicker">Avg Interest</div><h2>{df["interest_rate"].mean():.2%}</h2></div>', unsafe_allow_html=True)

st.markdown("---")

PLOTLY_TEMPLATE = "plotly_dark"

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.markdown("#### Risk Category Distribution")
    fig = px.pie(df, names="risk_category", hole=0.5, template=PLOTLY_TEMPLATE,
                 color_discrete_sequence=["#4FD37E", "#E4C766", "#FF7E8F"])
    st.plotly_chart(fig, use_container_width=True)

with row1_col2:
    st.markdown("#### Approval vs Rejection")
    fig = px.histogram(decided, x="decision", template=PLOTLY_TEMPLATE, color="decision",
                        color_discrete_map={"Approved": "#4FD37E", "Approved with Review": "#E4C766",
                                             "Rejected": "#FF7E8F"})
    st.plotly_chart(fig, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.markdown("#### CIBIL Score Distribution")
    fig = px.histogram(df, x="cibil_score", nbins=20, template=PLOTLY_TEMPLATE,
                        color_discrete_sequence=["#C9A648"])
    st.plotly_chart(fig, use_container_width=True)

with row2_col2:
    st.markdown("#### Loan Amount Histogram")
    fig = px.histogram(df, x="requested_loan_amount", nbins=20, template=PLOTLY_TEMPLATE,
                        color_discrete_sequence=["#5B8DEF"])
    st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Income vs Loan Amount")
fig = px.scatter(
    df, x="monthly_income", y="requested_loan_amount", color="risk_category",
    template=PLOTLY_TEMPLATE, color_discrete_map={"Low": "#4FD37E", "Medium": "#E4C766", "High": "#FF7E8F"},
    hover_data=["full_name", "cibil_score", "decision"],
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("#### All Applications")
st.dataframe(
    df[["id", "full_name", "cibil_score", "risk_category", "decision", "approved_loan_amount",
        "interest_rate", "created_at"]].sort_values("id", ascending=False),
    use_container_width=True,
)
