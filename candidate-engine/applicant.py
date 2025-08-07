"""
    This class contains the applicant's data.

    Attributes:
        name (string): name of the applicant
        resume (string): resume text of the applicant
        similarity (float): cosine similarity of the applicant
"""

class Applicant:
    def __init__(self, name, resume, similarity):
        self.name = name
        self.resume = resume
        self.similarity = similarity
