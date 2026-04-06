# ============================================================
# STEP 1: Install dependencies (run once in Colab)
# ============================================================
!pip install -q langchain langchain-community langchain-google-genai pypdf python-docx

# ============================================================
# STEP 2: Imports
# ============================================================
import os
import json
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# ============================================================
# STEP 3: Set API Key
# ============================================================
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"

# ============================================================
# STEP 4: File Loader Function (clean + reusable)
# ============================================================
def load_resume(file_path):
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()

    # Combine directly (no fake chunking nonsense)
    text = " ".join([doc.page_content for doc in docs]).strip()

    if not text:
        raise ValueError("❌ Failed to extract text from resume.")

    return text

# ============================================================
# STEP 5: Optional Chunking (ONLY if needed)
# ============================================================
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_text(text)

# ============================================================
# STEP 6: Define LLM
# ============================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.3
)

# ============================================================
# STEP 7: STRONG Prompt (Structured Output)
# ============================================================
prompt = PromptTemplate(
    input_variables=["job_desc", "resume_text"],
    template="""
You are an expert AI recruiter.

Analyze the candidate resume against the job description.

JOB DESCRIPTION:
{job_desc}

RESUME:
{resume_text}

Return STRICT JSON ONLY in this format:

{{
    "match_score": (0-100 integer),
    "matched_skills": [list of relevant matched skills],
    "missing_skills": [list of important missing skills],
    "experience_summary": "brief evaluation of candidate experience",
    "final_verdict": "Short decision: Strong Fit / Moderate Fit / Weak Fit"
}}

IMPORTANT:
- Do NOT add explanations outside JSON
- Be strict and realistic in scoring
"""
)

# ============================================================
# STEP 8: Create Chain (modern style)
# ============================================================
chain = prompt | llm

# ============================================================
# STEP 9: Run Analysis
# ============================================================
def analyze_resume(file_path, job_description):
    resume_text = load_resume(file_path)

    # Safety preview (debugging visibility)
    print("🔍 Resume Preview:", resume_text[:300], "\n")

    response = chain.invoke({
        "job_desc": job_description,
        "resume_text": resume_text
    })

    # Try parsing JSON safely
    try:
        result = json.loads(response.content)
    except:
        print("⚠️ Raw output (failed JSON parse):")
        print(response.content)
        return None

    return result

# ============================================================
# STEP 10: Example Usage
# ============================================================
job_description = """
We are hiring a Senior Python Developer with experience in:
Python, SQL, AWS/GCP/Azure, LangChain, LLM tools, APIs, and Data Analysis.
"""

resume_path = "/content/sample_resume.pdf"  # change this

result = analyze_resume(resume_path, job_description)

# ============================================================
# STEP 11: Output
# ============================================================
if result:
    print("\n✅ FINAL RESULT:")
    print(json.dumps(result, indent=4))
