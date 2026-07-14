"""
Project Matcher Platform

Matches interns/employees to open projects based on skills/interests vs. project
requirements, using TF-IDF + cosine similarity for ranking and Claude
for generating human-readable match explanations.

Workflow:
1. Officer/user uploads interns.csv and projects.csv
2. Hard filters remove ineligible matches (availability, open slots)
3. Remaining matches are ranked by text similarity
4. Top 3 matches per intern are shown with an AI-generated explanation of why they are a match
"""

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from anthropic import Anthropic
import gspread
from google.oauth2.service_account import Credentials

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

st.set_page_config(page_title="Project Matcher", layout="wide")
st.title("Project Matcher")

# Reads in live data from google sheets using Google Sheet API

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Try to load credentials from Streamlit's secrets system (used once deployed)
try:
    has_secrets = "gcp_service_account" in st.secrets
except Exception:
    has_secrets = False

if has_secrets:
    # Running on Streamlit Cloud
    creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=SCOPES)
else:
    creds = Credentials.from_service_account_file("service_account.json", scopes=SCOPES)
gc = gspread.authorize(creds)

INTERNS_SHEET_ID = os.getenv("INTERNS_SHEET_ID")
PROJECTS_SHEET_ID = os.getenv("PROJECTS_SHEET_ID")

st.subheader("Live Data")

if st.button("Refresh Data"):
    st.cache_data.clear()

@st.cache_data(ttl=60)
def load_data():
    interns_sheet = gc.open_by_key(INTERNS_SHEET_ID).sheet1
    projects_sheet = gc.open_by_key(PROJECTS_SHEET_ID).sheet1
    interns_df = pd.DataFrame(interns_sheet.get_all_records())
    projects_df = pd.DataFrame(projects_sheet.get_all_records())
    return interns_df, projects_df

interns_df, projects_df = load_data()

# Shows data

with st.expander("View raw data (interns & projects)"):
    st.write("Interns:")
    st.dataframe(interns_df)
    st.write("Projects:")
    st.dataframe(projects_df)


# Load the embedding model when the app starts
model = SentenceTransformer("all-MiniLM-L6-v2")

def build_intern_text(row):
    # Combine the intern's skills, interests, and experience into one text blob
    return f"{row['skills']} {row['interests']} {row['experience']}"

def build_project_text(row):
    # Merge the skills needed + description into one string the model can embed
    return f"{row['skills_needed']} {row['project_description']}"

def generate_explanation(intern, project_row, score):
    # Find overlapping words between intern's background and project's needs
    intern_words = set(build_intern_text(intern).lower().replace(",", "").split())
    project_words = set(build_project_text(project_row).lower().replace(",", "").split())
    overlap = [w for w in (intern_words & project_words) if len(w) > 3]

    # Describe match strength in plain language based on the score
    if score >= 0.6:
        strength = "This is a strong match"
    elif score >= 0.35:
        strength = "This is a solid match"
    else:
        strength = "This is a possible match, though a bit of a stretch"

    if overlap:
        skills_list = ", ".join(overlap)
        return f"{strength} — {intern['name']} has direct experience or interest in {skills_list}, which lines up well with what this project needs."
    else:
        return f"{strength} — {intern['name']}'s overall background and interests align with this project's focus, even without an exact skills overlap."

st.subheader("Matches")

# Only run the matching logic when this button is clicked
if st.button("Run Matching"):
    with st.spinner("Matching interns to projects..."):
        intern_embeddings = [model.encode(build_intern_text(row)) for _, row in interns_df.iterrows()]
        project_embeddings = [model.encode(build_project_text(row)) for _, row in projects_df.iterrows()]

        for i, intern in interns_df.iterrows():
            eligible_indices = [
                j for j, project in projects_df.iterrows()
                if project["slots_open"] > 0
                and project["weekly_time_commitment_hrs"] <= intern["availability_hrs_per_week"] + 1
            ]

            scored = [
                (projects_df.iloc[j]["name"], util.cos_sim(intern_embeddings[i], project_embeddings[j]).item())
                for j in eligible_indices
            ]

            top3 = sorted(scored, key=lambda x: x[1], reverse=True)[:3]

            st.markdown(f"### {intern['name']}")

            if int(intern["agile_comfort"]) <= 2:
                st.warning(f"{intern['name']} rated their comfort working remote as {intern['agile_comfort']} out of 5.")

            has_shown_match = False
            for name, score in top3:
                percent = round(score * 100)
                if percent < 50:
                    continue
                has_shown_match = True
                project_row = projects_df[projects_df["name"] == name].iloc[0]
                reason = generate_explanation(intern, project_row, score)
                st.markdown(f"**{name}** — {percent}% match")
                st.write(reason)
                st.progress(min(score, 1.0))

            if not has_shown_match:
                st.info("No strong matches (50%+) found for this intern.")

            st.divider()