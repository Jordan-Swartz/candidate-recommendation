"""
    This file contains the utility functions for:
        - Parsing PDFs into raw tuple data
        - Generating vector embeddings
        - Calculating cosine similarity
        - Generating candidate summary
"""
import fitz
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Import Embedding Model and text model
model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
MODEL_ID = "google/flan-t5-small"
summarizer = pipeline("text2text-generation", model=MODEL_ID, device=-1)

def parse_resumes(uploaded_files):
    """
    Parses the resume PDFs and returns a list of raw text data tuples (filename, text)
    """
    candidates_text = []
    for uploaded_file in uploaded_files:
        try:
            #start from beginning of file and convert pdf to text doc
            uploaded_file.seek(0)
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                text = "\n".join(page.get_text() for page in doc)
            #add tuple to list
            candidates_text.append((uploaded_file.name, text))

        except Exception as e:
            print(f"Error processing {getattr(uploaded_file, 'name', '')}: {e}")
            candidates_text.append((uploaded_file.name, ""))

    return candidates_text

def generate_embedding(text):
    """
    Generates a vector embedding from the given text
    """
    return model.encode(text)

def calculate_similarity(job_description_embedding, applicant_embedding):
    """
    Calculates the cosine similarity between two vector embeddings.
    Reshapes the embeddings into 2D arrays
    """
    job_emb = np.array(job_description_embedding).reshape(1, -1)
    app_emb = np.array(applicant_embedding).reshape(1, -1)
    return cosine_similarity(job_emb, app_emb)[0][0]

sum_resume = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
sum_jd     = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

fit = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

def compress(text: str, max_chars=4000):
    text = (text or "")[:max_chars]
    return sum_resume(text, max_length=120, min_length=60, do_sample=False)[0]["summary_text"]

def generate_applicant_summary(job_description: str, resume_text: str) -> str:
    jd_sum  = compress(job_description)
    res_sum = compress(resume_text)

    prompt = (
        "In exactly two sentences, assess the candidate's fit using overlap between the job and resume. "
        "Mention concrete skills/tools. End with 'Fit: strong', 'Fit: medium', or 'Fit: weak'.\n\n"
        f"Job (summary): {jd_sum}\n"
        f"Resume (summary): {res_sum}\n\n"
        "Summary:"
    )
    return fit(prompt, max_new_tokens=60, do_sample=False, num_beams=4, truncation=True)[0]["generated_text"].strip()