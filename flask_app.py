import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------
# ENV & APP SETUP
# --------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__, template_folder="templates")

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
# ROUTES
# --------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        brand_name = request.form.get("brand_name", "Your Brand")
        file = request.files.get("file")

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        df = pd.read_csv(file)
        df_clean = load_campaign_data(df)
        summary = summarize_data(df_clean)
        insights = analyze_with_llm(df_clean, summary, brand_name)

        return jsonify({"summary": summary, "insights": insights})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
