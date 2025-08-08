"""
    This file contains the utility functions for:
        - Parsing PDFs into raw tuple data
        - Generating vector embeddings
        - Calculating cosine similarity
        - Generating candidate summary
"""
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import Embedding Model
model = SentenceTransformer('all-MiniLM-L6-v2')

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

def generate_applicant_summary(job_description, applicant):
    """
    Generates a summary of the applicant's fit for the given job description
    """
    return applicant.summary