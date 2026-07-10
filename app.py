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

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

st.set_page_config(page_title="Project Matcher", layout="wide")
st.title("Project Matcher")