"""
    This class contains the UI layout for the application
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
from candidate_engine.utils import *
from candidate_engine.applicant import Applicant

# collect resume & name
# calculate cosine similarity
# create applicant object
# sort list

# Sidebar
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Header & Info
with st.container():
    st.markdown(
        """
        <div style="
            background-color: #2196f3;
            border-radius: 15px;
            margin-bottom: 1rem;
            color: white;
            text-align: center;
        ">
            <h1 style="margin-bottom: 0.3em; font-size: 3rem;">
                Candidate Recommendation Engine
            </h1>
            <h3 style='font-style: italic; font-weight: normal;'>
                Find The Best Candidate For The Job!
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

# How to info
st.subheader("How to use")
#row 1
col_icon, col_text = st.columns([1, 8])
with col_icon:
    st.image("resources/img/upload.png", width=48)
with col_text:
    st.markdown("Get started by entering your desired job description into the provided text field. Then, upload your candidate resumes either by pasting the text directly or by uploading files—whichever works best for you!")
st.divider()

#row 2
col_icon, col_text = st.columns([1, 8])
with col_icon:
    st.image("resources/img/brain.png", width=48)
with col_text:
    st.markdown("Allow our engine to do the heavy lifting by analyzing the job description and resumes using advanced natural language processing. It will compute the cosine similarity between the job requirements and each resume to identify the most relevant candidates!")
st.divider()

#row 3
col_icon, col_text = st.columns([1, 8])
with col_icon:
    st.image("resources/img/bulb.png", width=52)
with col_text:
    st.markdown("After processing, enjoy a ranked list of candidates based on how well they match the job description! Our engine provides a detailed breakdown of each match, helping you understand exactly why a candidate received their ranking.")

st.markdown(
            "<h4 style='text-align: center;'>Get Started Below!</h4>",
            unsafe_allow_html=True
        )
st.divider()

# Collect Input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Enter Job Description")
    job_description = st.text_area("Job Description", height=160)

with col2:
    st.subheader("Upload Resume Files")
    text_toggle_state = st.session_state.get('text_toggle_state')

    if text_toggle_state:
        uploaded_resume_text = st.text_area("Resume Input", height=160)
    else:
        uploaded_resumes = st.file_uploader("Upload Resume PDFs", accept_multiple_files=True)

    text_toggle = st.toggle('Text Input', value=text_toggle_state, key="text_toggle_state")

#TODO:  raw_resume_data for text inputs list manually

# Engine Computation
cols = st.columns([1, 4, 1])
with cols[1]:
    start = st.button("Recommend Candidates", use_container_width=True)

    if not job_description or not uploaded_resumes:
        st.warning("Please provide a job description and at least one resume.")

    if text_toggle is False:
        #process PDF list into raw text
        raw_resume_data = parse_resumes(uploaded_resumes)

    #create applicant objs from raw text
    applicants = []
    for filename, resume_text in raw_resume_data:
        name = filename.rsplit('.', 1)[0]
        applicant = Applicant(name, resume_text)
        applicants.append(applicant)

    #generate job description embedding
    job_description_embedding = generate_embedding(job_description)

    #generate vector embedding and compute cosine similarity for each applicant
    for applicant in applicants:
        applicant.embedding = generate_embedding(applicant.resume_text)
        applicant.similarity = calculate_similarity(job_description_embedding, applicant.embedding)
    #sort applicants by cosine similarity (descending: best to worst)
    applicants.sort(key=lambda applicant: applicant.similarity, reverse=True)

    #generate summaries for top 10 applicants
    # for applicant in applicants[:10]:
        #applicant.summary = generate_applicant_summary()

# Display Engine Results
st.markdown("### Top Candidates")
for idx, applicant in enumerate(applicants[:10], 1):
    with st.container():
        st.markdown(
            "<div style='background-color:#f4f8fd; border-radius:12px; padding:1.2em 1em; margin-bottom:0.9em; border:1px solid #dbe7fa;'>",
            unsafe_allow_html=True,
        )

        # -- YOUR COLUMNS/TABLE GOES HERE! --
        col1, col2, col3, col4 = st.columns([2, 2, 2, 8])
        col1.markdown("**Rank**")
        col2.markdown("**Name**")
        col3.markdown("**Similarity**")
        col4.markdown("**Summary**")
        col1.markdown(f"{idx}")
        col2.markdown(f"{applicant.name}")
        col3.markdown(f"{applicant.similarity:.2f}")
        col4.markdown(f"{applicant.summary or 'No summary generated.'}")

        st.markdown("</div>", unsafe_allow_html=True)

# Download
top_applicants = [
    {
        "Rank": idx+1,
        "Name": applicant.name,
        "Similarity": round(applicant.similarity, 3),
        "Summary": applicant.summary or "No summary generated."
    }
    for idx, applicant in enumerate(applicants[:10])
]
df = pd.DataFrame(top_applicants)

cols = st.columns([1, 1, 1])
with cols[1]:
    st.download_button(
        label="Download Results as CSV",
        data=df.to_csv(index=False),
        file_name="top_candidates.csv",
        mime="text/csv",
)

st.divider()

# Feedback Footer
cols = st.columns([1, 1, 1])
with cols[1]:
    st.markdown("*Give us a rating!*")
    sentiment_mapping = ["one", "two", "three", "four", "five"]
    selected = st.feedback("stars")
    if selected is not None:
        st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")
