# ⚙️ Candidate Recommendation Engine ⚙️
A candidate engine in Python that analyzes and ranks the most relevant job candidates given their resumes and a job description.

---

## 1: 📌 About the Project:
A Python and Streamlit-based web application that recommends the most relevant job candidates based on a provided job description. The engine uses modern NLP techniques to generate semantic embeddings of resumes and job descriptions, then computes similarity scores to rank candidates. 

---

## 2: 🔍 Featured Concepts:
1. Semantic Search with Vector Embeddings
   - Generates vector embeddings for the job description and resumes using `sentence-transformers`. This allows for the text to be analyzed contextually.
2. Cosine Similarity Ranking
   - Uses `scikit-learn` to compute similarity scores between the job description and each candidate vector.
   - Produces a descending ranking from best to worst similarity score.
3. AI Fit Summaries
   - Uses Hugging Face `transformers pipeline` to generate a brief summary of why the candidate is a good fit.
4. Flexible Input Options
   - Accepts resumes as a text input or PDF file upload.
   - Accepts job description as a text input.
   - Uses `pymupdf` to parse PDFs.
5. Streamlit-Powered UI
   - Simple, responsive, interactive, and modular UI.
   - Allows the user to choose how many candidates they want to display (1-20).
    
---

## 3: 🚀 Running the Project:
### 🔗 Public Link
- Streamlit Link: [Project Link](https://jordan-swartz-candidate-recommendation-app-qmui0u.streamlit.app/)
### ⬆️ Project Setup
1. Create Virtual Environment
   - `python3 -m venv venv`
     
2. Install dependencies
   - `pip install -r requirements.txt`

3. Run the Streamlit App
   - `streamlit run app.py`

### 🎥 Project Demo
- Screencast: [Project Demo](https://youtu.be/Cc4Vt-tlOio)
---

## 4: 🛠 Technologies Used:
1. Python 3.9.6
   - core programming language
2. Streamlit
   - frontend and deployment
3. Sentence-Transformers
   - embedding generation
4. Scikit-learn
   - cosine similar calculation
5. PyMuPDF
   - PDF parsing
6. Hugging Face Transformers
   - AI Summaries
  
---

## 5: 📚 Resources Used:
1. DataStax
   - [Vector Embeddings](https://www.datastax.com/blog/how-to-create-vector-embeddings-in-python) - Creating embeddings in Python
2. GeeksforGeeks
   - [Cosine Similarity](https://www.geeksforgeeks.org/dbms/cosine-similarity/) - Understanding cosine similarity
3. Streamlit
   - [Documentation](https://docs.streamlit.io/) - Understanding Streamlit library
4. PyMuPDF
   - [PDF Parsing](https://pymupdf.readthedocs.io/en/latest/) - Understanding PyMuPDF installation and implementation
5. Hugging Face
   - [Pipeline Documentation](https://huggingface.co/docs/transformers/pipeline_tutorial) - Understanding Pipeline installation and implementation
     
---


