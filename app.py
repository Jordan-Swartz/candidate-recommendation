import streamlit as st
import pandas as pd
from candidate_engine.utils import *
from candidate_engine.applicant import Applicant

# Sidebar
add_selectbox = st.sidebar.selectbox(
    'Account Login - *Coming Soon*',
    ('Email', 'Home phone', 'Mobile phone')
)

# Header & Info
with st.container():
    st.markdown(
        """
        <div style="
            background-color: #172A4F;
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

# How To Info
st.subheader("How To Use")
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
    st.markdown("After processing, enjoy a ranked list of candidates based on how well they match the job description! Our engine provides an optional detailed breakdown of each match, helping you understand exactly why a candidate received their ranking.")
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

    ss = st.session_state
    ss.setdefault("text_toggle_state", False)  #default PDF
    ss.setdefault("prev_text_toggle_state", ss.text_toggle_state)
    ss.setdefault("manual_resumes", [])
    ss.setdefault("resume_text_area", "")
    ss.setdefault("clear_resume_text", False)

    uploaded_resumes = []

    if ss.text_toggle_state:
        #text mode
        with st.form("manual_resume_form", clear_on_submit=True):
            resume_input = st.text_area("Resume Input", key="resume_text_area", height=160)
            add_clicked = st.form_submit_button("Add Resume")

            if add_clicked and resume_input.strip():
                ss.manual_resumes.append(resume_input.strip())

        if ss.manual_resumes:
            st.markdown("##### Added Resumes:")
            for idx, res in enumerate(ss.manual_resumes, 1):
                st.markdown(f"- **Resume {idx}** ({len(res)} chars)")

        with st.container():
            col1, col2 = st.columns([1, 1])

            with col1:
                text_toggle_state = st.toggle("Text Input", key="text_toggle_state", value=False)
            with col2:
                summary_toggle_state = st.toggle("AI Summary", key="summary_toggle_state", value=False)

    else:
        #pdf mode
        uploaded_resumes = st.file_uploader("Upload Resume PDFs", accept_multiple_files=True)
        with st.container():
            col1, col2 = st.columns([1, 1])

            with col1:
                text_toggle_state = st.toggle("Text Input", key="text_toggle_state", value=False)
            with col2:
                summary_toggle_state = st.toggle("AI Summary", key="summary_toggle_state", value=False)
        # st.toggle("Text Input", key="text_toggle_state")

    #clear list when activating toggle
    if ss.prev_text_toggle_state and not ss.text_toggle_state:
        ss.manual_resumes = []
        ss.resume_text_area = ""
        ss.clear_resume_text = False

    ss.prev_text_toggle_state = ss.text_toggle_state

#expose for engine
text_toggle_state = st.session_state.text_toggle_state

# Engine Computation
#store states
if "applicants" not in st.session_state:
    st.session_state.applicants = []
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

with st.container():
    start = st.button("Recommend Candidates")

    if start:
        # validate by mode
        no_resumes = (
                (text_toggle_state is False and not uploaded_resumes) or
                (text_toggle_state is True and not st.session_state.manual_resumes)
        )

        if not job_description or no_resumes:
            st.warning("Please provide a job description and at least one resume.")
        else:
            with st.spinner("Processing candidates..."):
                if text_toggle_state is False:
                    #process PDF list into raw text
                    raw_resume_data = parse_resumes(uploaded_resumes)
                else:
                    raw_resume_data = [(f"manual_{i + 1}.txt", txt) for i, txt in
                                       enumerate(st.session_state.manual_resumes)]

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

                #generate summaries for applicants
                for applicant in applicants:
                    if summary_toggle_state:
                        applicant.summary = generate_applicant_summary(job_description, applicant.resume_text)
                    else:
                        applicant.summary = "No Summary Provided."

                st.session_state.applicants = applicants
                st.session_state.results_ready = True

            st.success("Candidate recommendations generated!")

# Display Engine Results
if st.session_state.results_ready:
    applicants = st.session_state.applicants

    st.divider()
    st.subheader("Top Candidates")

    #candidate cards
    num = st.slider(
        "*Select The Number of Candidates to Display*",
        min_value=1, max_value=min(20, len(applicants)), value=5
    )

    card_bg = "#172A4F"
    header_bg = "#223A6A"
    #table header
    st.markdown(
        f"""
        <div style="background-color: {header_bg}; border-radius: 10px 10px 0 0; padding: 1em; margin-bottom: 0.8em; color: #fff; font-weight: bold;">
            <div style="display: flex;">
                <div style="flex:1;">Rank</div>
                <div style="flex:2;">Name</div>
                <div style="flex:2;">Similarity</div>
                <div style="flex:6;">Summary</div>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    for idx, applicant in enumerate(applicants[:num], 1):
        st.markdown(
            f"""
            <div style="background-color: {card_bg}; border-radius: 0 0 10px 10px; padding: 1.1em 1em; margin-bottom: 0.8em; border: 1.5px solid #223A6A;">
                <div style="display: flex;">
                    <div style="flex:1; font-weight: bold;">#{idx}</div>
                    <div style="flex:2; font-weight: bold;">{applicant.name}</div>
                    <div style="flex:2;">{applicant.similarity:.2f}</div>
                    <div style="flex:6;">{applicant.summary or 'No summary generated.'}</div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

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

    with st.container():
        st.download_button(
            label="Download Results as CSV",
            data=df.to_csv(index=False),
            file_name="top_candidates.csv",
            mime="text/csv",
        )

st.divider()


# Feedback Footer
st.subheader("Give Us a Rating!")
with st.container():
    selected = st.feedback("stars")
    if selected is not None:
        st.markdown("*Thank You!*")
