"""
    This class contains the UI layout for the application
"""
import streamlit as st
from candidate_engine.resume_parser import parse_resumes
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
    st.image("resources/upload.png", width=48)
with col_text:
    st.markdown("Get started by entering your desired job description into the provided text field. Then, upload your candidate resumes either by pasting the text directly or by uploading files—whichever works best for you!")
st.divider()

#row 2
col_icon, col_text = st.columns([1, 8])
with col_icon:
    st.image("resources/brain.png", width=48)
with col_text:
    st.markdown("Allow our engine to do the heavy lifting by analyzing the job description and resumes using advanced natural language processing. It will compute the cosine similarity between the job requirements and each resume to identify the most relevant candidates!")
st.divider()

#row 3
col_icon, col_text = st.columns([1, 8])
with col_icon:
    st.image("resources/bulb.png", width=52)
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

# Begin Computation
cols = st.columns([2, 2, 2])
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

    #generate vector embeddings for each applicant

