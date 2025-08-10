import PyPDF2
import docx

def extract_text_from_file(path):
    if path.endswith(".pdf"):
        return extract_from_pdf(path)
    elif path.endswith(".docx"):
        return extract_from_docx(path)
    else:
        return ""

def extract_from_pdf(path):
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])
