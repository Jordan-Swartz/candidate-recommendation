"""
    This class contains the applicant's data.
    Attributes:
        name (string): name of the applicant
        resume (string): resume text of the applicant
        embedding (list of float): vector embedding of the resume
        similarity (float): cosine similarity of the applicant
        summary (string): summary of the applicant
"""

class Applicant:
    def __init__(self, name, resume_text):
        self.name = name
        self.resume_text = resume_text
        self.embedding = None
        self.similarity = None
        self.summary = None
