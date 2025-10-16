import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------
# ENV SETUP
# --------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AdAuto Brand Analyzer", page_icon="📊", layout="wide")
st.title("📊 AdAuto — AI Brand Campaign Analyzer")
st.caption("Upload your ad campaign CSV and get instant AI-powered brand insights.")

# --------------------------
# HELPER FUNCTIONS
# --------------------------
def load_campaign_data(df: pd.DataFrame) -> pd.DataFrame:
    def _norm_col_name(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    col_map = {_norm_col_name(c): c for c in df.columns}

    def find_col(aliases):
        for a in aliases:
            key = _norm_col_name(a)
            if key in col_map:
                return col_map[key]
        return None

    mappings = {
        "Campaign": ["campaign", "campaign name"],
        "Clicks": ["clicks"],
        "Impressions": ["impressions", "imps"],
        "Cost": ["cost", "spend"],
        "Conversions": ["conversions"],
        "Ctr": ["ctr", "ctr%", "clickthroughrate"]
    }

    rename_map = {}
    for canon, aliases in mappings.items():
        found = find_col(aliases)
        if found:
            rename_map[found] = canon
    df = df.rename(columns=rename_map)

    df.fillna(0, inplace=True)

    def to_numeric_col(col):
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    for col in ["Clicks", "Impressions", "Cost", "Conversions", "Ctr"]:
        to_numeric_col(col)
        if col not in df.columns:
            df[col] = 0

    if "Ctr" not in df.columns or df["Ctr"].sum() == 0:
        if "Clicks" in df.columns and "Impressions" in df.columns and df["Impressions"].sum() > 0:
            df["Ctr"] = (df["Clicks"] / df["Impressions"]) * 100
        else:
            df["Ctr"] = 0

    if "Campaign" not in df.columns:
        df["Campaign"] = "(unknown)"

    return df


def summarize_data(df: pd.DataFrame) -> dict:
    summary = {
        "Total Campaigns": len(df["Campaign"].unique()),
        "Total Ads": len(df),
        "Avg CTR (%)": round(df["Ctr"].mean(), 2),
        "Avg CPC ($)": round(df["Cost"].sum() / max(df["Clicks"].sum(), 1), 2),
        "Avg Conversion Rate (%)": round(df["Conversions"].sum() / max(df["Clicks"].sum(), 1) * 100, 2),
    }
    return summary


def analyze_with_llm(df: pd.DataFrame, summary: dict, brand_name: str) -> str:
    data_text = df.to_csv(index=False)
    prompt = f"""
You are AdAuto — an AI Brand Analysis engine.
Analyze the following campaign data for brand "{brand_name}".
Generate a structured insight covering:

1. Campaign Highlights
2. Strengths
3. Weaknesses
4. AI-Powered Recommendations
5. Tone and Creative Analysis
6. Brand Positioning Summary

Summary metrics:
{summary}

Dataset:
{data_text}

Write insights in a concise, professional, data-backed format.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert marketing analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
    )
    return response.choices[0].message.content.strip()


# --------------------------
# STREAMLIT UI
# --------------------------
st.markdown("---")

brand_name = st.text_input("🏷️ Enter Brand Name", "Digital Piloto")
uploaded_file = st.file_uploader("📂 Upload Your CSV File", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

        st.subheader("📋 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        df_clean = load_campaign_data(df)
        summary = summarize_data(df_clean)

        st.subheader("📈 Campaign Summary Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Campaigns", summary["Total Campaigns"])
        col2.metric("Total Ads", summary["Total Ads"])
        col3.metric("Avg CTR (%)", summary["Avg CTR (%)"])

        col4, col5 = st.columns(2)
        col4.metric("Avg CPC ($)", summary["Avg CPC ($)"])
        col5.metric("Avg Conversion Rate (%)", summary["Avg Conversion Rate (%)"])

        if st.button("🚀 Generate AI Insights"):
            with st.spinner("Analyzing your campaigns using AdAuto AI..."):
                insights = analyze_with_llm(df_clean, summary, brand_name)
            st.success("✅ AI Analysis Complete!")
            st.subheader("🧠 AI-Generated Brand Insights")
            st.write(insights)

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👆 Please upload a CSV file to begin analysis.")
