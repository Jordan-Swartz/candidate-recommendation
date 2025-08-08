import fitz

"""
    This file contains the logic for parsing the resume PDFs
"""

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
