import re
import PyPDF2
import os
import spacy
import subprocess
import logging
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Configure logging for debugging on Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Load spaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully")
except OSError:
    logger.info("Downloading spaCy model 'en_core_web_sm'...")
    try:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model 'en_core_web_sm' downloaded and loaded successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download spaCy model: {str(e)}")
        raise Exception("Failed to download spaCy model")

# === Text Extraction Functions ===
def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Extracted text in lowercase, or empty string on failure.
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        logger.debug(f"Extracted text from PDF: {pdf_path}")
        return text.lower().strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return ""

def extract_text_from_docx(docx_path):
    """
    Extract text from a DOCX file.
    Args:
        docx_path (str): Path to the DOCX file.
    Returns:
        str: Extracted text in lowercase, or empty string on failure.
    """
    try:
        doc = Document(docx_path)
        text = "\n".join([para.text.lower() for para in doc.paragraphs if para.text.strip()])
        logger.debug(f"Extracted text from DOCX: {docx_path}")
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {docx_path}: {str(e)}")
        return ""

# === ATS Compatibility Check ===
def check_ats_compatibility(resume_text, job_description=""):
    """
    Check the ATS compatibility of a resume.
    Args:
        resume_text (str): Extracted text from the resume.
        job_description (str): Job description text for keyword matching.
    Returns:
        dict: Analysis results including score, warnings, recommendations, and details.
    """
    logger.info("Starting ATS compatibility check")
    if not resume_text:
        logger.error("Resume text is empty")
        return {
            "score": 0,
            "warnings": ["No text extracted from resume"],
            "recommendations": ["Ensure the resume is machine-readable (avoid scanned PDFs)"],
            "details": {}
        }

    results = {
        "score": 100,
        "warnings": [],
        "recommendations": [],
        "details": {
            "structure": {},
            "content": {},
            "formatting": {},
            "keywords": {}
        }
    }

    # 1. Check for required sections
    required_sections = [
        ("contact", r"contact|details|information"),
        ("summary", r"summary|objective|profile"),
        ("experience", r"experience|work history|employment"),
        ("education", r"education|academic background|qualifications"),
        ("skills", r"skills|technical skills|competencies"),
        ("projects", r"projects|portfolio|achievements"),
        ("certifications", r"certifications|licenses|accreditations")
    ]

    found_sections = []
    missing_sections = []
    section_positions = {}
    for section, pattern in required_sections:
        match = re.search(pattern, resume_text, re.IGNORECASE)
        if match:
            found_sections.append(section)
            section_positions[section] = match.start()
        else:
            missing_sections.append(section)

    results["details"]["structure"]["found_sections"] = found_sections
    results["details"]["structure"]["missing_sections"] = missing_sections

    if missing_sections:
        results["warnings"].append(f"Missing key sections: {', '.join(missing_sections)}")
        results["score"] -= len(missing_sections) * 5

    # 2. Check section order
    if len(section_positions) > 1:
        ordered_sections = [k for k, v in sorted(section_positions.items(), key=lambda item: item[1])]
        results["details"]["structure"]["section_order"] = ordered_sections
        if "experience" in ordered_sections and "education" in ordered_sections:
            exp_idx = ordered_sections.index("experience")
            edu_idx = ordered_sections.index("education")
            if edu_idx < exp_idx:
                results["recommendations"].append("Consider placing Experience before Education (unless a recent graduate)")
                results["score"] -= 3

    # 3. Check contact information
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', resume_text)
    phones = re.findall(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', resume_text)
    has_linkedin = bool(re.search(r'linkedin\.com', resume_text))

    results["details"]["content"]["contact_info"] = {
        "emails": emails,
        "phones": phones,
        "linkedin": has_linkedin
    }

    if not emails:
        results["warnings"].append("Missing email address")
        results["score"] -= 5
    if not phones:
        results["warnings"].append("Missing phone number")
        results["score"] -= 5
    if not has_linkedin:
        results["recommendations"].append("Consider adding a LinkedIn profile URL")

    # 4. Check date format consistency
    date_patterns = [
        (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b', "month_year"),
        (r'\b\d{2}/\d{4}\b', "mm_yyyy"),
        (r'\b\d{2}/\d{2}/\d{4}\b', "mm_dd_yyyy"),
        (r'\b(19|20)\d{2}\b', "yyyy")
    ]

    found_date_formats = {}
    for pattern, fmt in date_patterns:
        matches = re.findall(pattern, resume_text)
        if matches:
            found_date_formats[fmt] = len(matches)

    results["details"]["formatting"]["date_formats"] = found_date_formats
    if len(found_date_formats) > 1:
        results["warnings"].append("Inconsistent date formats detected")
        results["score"] -= 3
    elif not found_date_formats:
        results["warnings"].append("No standard date formats found")
        results["score"] -= 5

    # 5. Check for action verbs
    action_verbs = [
        "developed", "managed", "led", "designed", "implemented", "improved", "analyzed", "created",
        "coordinated", "executed", "facilitated", "generated", "increased", "negotiated", "resolved",
        "streamlined", "supervised", "trained", "optimized", "delivered"
    ]

    found_verbs = [verb for verb in action_verbs if re.search(r'\b' + verb + r'\b', resume_text)]
    results["details"]["content"]["action_verbs"] = {
        "found": found_verbs,
        "count": len(found_verbs),
        "ratio": len(found_verbs) / len(action_verbs)
    }

    if len(found_verbs) < 5:
        missing_verbs = [verb for verb in action_verbs if verb not in found_verbs]
        results["recommendations"].append(f"Add more action verbs, e.g., {', '.join(missing_verbs[:3])}")
        results["score"] -= 5

    # 6. Check formatting issues (bullet points, spacing, etc.)
    bullet_patterns = [r'•\s', r'-\s', r'\*\s', r'✓\s', r'➢\s']
    bullet_types = {pattern: len(re.findall(pattern, resume_text)) for pattern in bullet_patterns if len(re.findall(pattern, resume_text)) > 0}
    results["details"]["formatting"]["bullet_types"] = bullet_types
    if len(bullet_types) > 1:
        results["warnings"].append("Inconsistent bullet point styles")
        results["score"] -= 3

    double_spaces = len(re.findall(r'\s\s+', resume_text))
    results["details"]["formatting"]["double_spaces"] = double_spaces
    if double_spaces > 5:
        results["warnings"].append("Multiple double spaces detected")
        results["score"] -= 2

    # 7. Check for ATS-unfriendly elements
    formatting_issues = []
    if re.search(r'table|column|graphic|image', resume_text):
        formatting_issues.append("tables/columns/graphics")
        results["warnings"].append("Avoid using tables, columns, or graphics")
        results["score"] -= 5

    hyperlinks = re.findall(r'https?://\S+', resume_text)
    if hyperlinks:
        formatting_issues.append("hyperlinks")
        results["warnings"].append("Avoid hyperlinks; they may not parse correctly")
        results["score"] -= 3
    results["details"]["formatting"]["issues"] = formatting_issues

    # 8. Check keyword alignment with job description
    if job_description:
        similarity_score = calculate_tfidf_similarity(resume_text, job_description)
        results["details"]["keywords"]["similarity_score"] = round(similarity_score * 100, 2)
        if similarity_score < 0.4:
            results["score"] -= 10
            results["recommendations"].append(f"Low keyword alignment with job description ({round(similarity_score * 100, 2)}%)")
            missing_keywords = extract_missing_keywords(resume_text, job_description)
            if missing_keywords:
                results["details"]["keywords"]["missing_keywords"] = missing_keywords
                results["recommendations"].append(f"Consider adding keywords: {', '.join(missing_keywords[:5])}")

    # 9. Check content quality
    word_count = len(re.findall(r'\b\w+\b', resume_text))
    results["details"]["content"]["word_count"] = word_count
    if word_count < 300:
        results["warnings"].append("Resume is too short (< 300 words)")
        results["score"] -= 10
    elif word_count > 1000:
        results["warnings"].append("Resume is too long (> 1000 words)")
        results["score"] -= 5

    pronouns = ["i ", " me ", " my ", " mine ", " myself "]
    pronoun_count = sum(resume_text.count(pronoun) for pronoun in pronouns)
    results["details"]["content"]["pronoun_count"] = pronoun_count
    if pronoun_count > 3:
        results["recommendations"].append("Avoid personal pronouns (e.g., I, me, my)")
        results["score"] -= 3

    quantifiables = len(re.findall(r'\d+%|\$\d+|\d+\s*(years?|months?|projects?|clients?)', resume_text))
    results["details"]["content"]["quantifiables"] = quantifiables
    if quantifiables < 3:
        results["recommendations"].append("Add more quantifiable achievements (e.g., increased sales by 20%)")
        results["score"] -= 5

    # 10. Check skills section
    if "skills" in found_sections:
        skills_match = re.search(r'skills.*?(?:experience|education|projects|$)', resume_text, re.IGNORECASE | re.DOTALL)
        if skills_match:
            skills_text = skills_match.group(0)
            skills_list = extract_skills(skills_text)
            results["details"]["content"]["skills"] = skills_list
            if len(skills_list) < 5:
                results["recommendations"].append("Add more specific skills to the Skills section")
                results["score"] -= 5

    # 11. Cap the score
    results["score"] = max(0, min(100, round(results["score"])))
    logger.info(f"ATS compatibility check completed. Score: {results['score']}")
    return results

# === Helper Functions ===
def extract_skills(text):
    """
    Extract skills from text using predefined list and spaCy.
    Args:
        text (str): Text to extract skills from.
    Returns:
        list: List of identified skills.
    """
    predefined_skills = [
        "python", "java", "javascript", "sql", "html", "css", "react", "node.js", "aws", "azure",
        "docker", "kubernetes", "git", "excel", "tableau", "machine learning", "data analysis",
        "project management", "agile", "scrum", "communication", "leadership", "teamwork"
    ]

    skills = set()
    for skill in predefined_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            skills.add(skill)

    doc = nlp(text)
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()
        if 2 <= len(chunk_text.split()) <= 4 and chunk_text not in skills:
            skills.add(chunk_text)

    skills_list = sorted(list(skills))
    logger.debug(f"Extracted skills: {skills_list}")
    return skills_list

def extract_missing_keywords(resume_text, job_description):
    """
    Identify keywords present in job description but missing in resume.
    Args:
        resume_text (str): Resume text.
        job_description (str): Job description text.
    Returns:
        list: List of missing keywords.
    """
    job_doc = nlp(job_description.lower())
    job_keywords = set()
    for token in job_doc:
        if token.pos_ in ["NOUN", "PROPN", "VERB"] and len(token.text) > 3:
            job_keywords.add(token.text)
    for chunk in job_doc.noun_chunks:
        if 1 < len(chunk.text.split()) <= 3:
            job_keywords.add(chunk.text)

    resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
    missing_keywords = [kw for kw in job_keywords if kw not in resume_words and len(kw) > 3]
    logger.debug(f"Missing keywords: {missing_keywords}")
    return sorted(missing_keywords)

def calculate_tfidf_similarity(resume_text, job_description):
    """
    Calculate TF-IDF similarity between resume and job description.
    Args:
        resume_text (str): Resume text.
        job_description (str): Job description text.
    Returns:
        float: Similarity score between 0 and 1.
    """
    try:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        logger.debug(f"TF-IDF similarity score: {similarity}")
        return similarity
    except Exception as e:
        logger.error(f"Error calculating TF-IDF similarity: {str(e)}")
        return 0.0

# === Industry Benchmarking ===
def benchmark_against_industry(resume_text, industry=""):
    """
    Benchmark resume against industry-specific keywords.
    Args:
        resume_text (str): Resume text.
        industry (str): Specified industry (optional).
    Returns:
        dict: Benchmark results.
    """
    industry_keywords = {
        "software": ["python", "java", "javascript", "sql", "react", "aws", "docker", "git", "agile", "scrum"],
        "finance": ["accounting", "finance", "excel", "analysis", "investment", "banking", "audit", "tax"],
        "marketing": ["marketing", "seo", "content", "social media", "campaign", "analytics", "branding"],
        "healthcare": ["patient", "clinical", "medical", "healthcare", "nursing", "hospital", "emr"],
        "data": ["data", "sql", "python", "tableau", "machine learning", "statistics", "analytics"]
    }

    if not industry:
        max_matches = 0
        detected_industry = "general"
        for ind, keywords in industry_keywords.items():
            matches = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', resume_text, re.IGNORECASE))
            if matches > max_matches:
                max_matches = matches
                detected_industry = ind
        industry = detected_industry

    keywords = industry_keywords.get(industry.lower(), [])
    present_keywords = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', resume_text, re.IGNORECASE)]
    missing_keywords = [kw for kw in keywords if kw not in present_keywords]

    relevance = len(present_keywords) / len(keywords) if keywords else 0
    benchmark = {
        "industry": industry,
        "industry_relevance": round(relevance, 2),
        "present_keywords": present_keywords,
        "missing_keywords": missing_keywords
    }
    logger.debug(f"Industry benchmark: {benchmark}")
    return benchmark

# === File Format Check ===
def check_file_format(file_path):
    """
    Check file format and size for ATS compatibility.
    Args:
        file_path (str): Path to the resume file.
    Returns:
        dict: File format analysis.
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    issues = []

    if file_ext not in [".pdf", ".docx"]:
        issues.append(f"Unsupported file format '{file_ext}'. Use PDF or DOCX")

    if file_size_mb > 5:
        issues.append(f"File size ({file_size_mb:.2f} MB) exceeds 5 MB")

    if file_ext == ".pdf":
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = reader.pages[0].extract_text() if reader.pages else ""
                if not text or len(text.strip()) < 50:
                    issues.append("PDF may not be machine-readable (e.g., scanned document)")
        except Exception as e:
            issues.append(f"Error reading PDF: {str(e)}")

    result = {
        "format": file_ext,
        "size_mb": round(file_size_mb, 2),
        "issues": issues
    }
    logger.debug(f"File format check: {result}")
    return result

# === Report Generation ===
def generate_report(analysis):
    """
    Generate a detailed ATS report as a string.
    Args:
        analysis (dict): ATS compatibility analysis results.
    Returns:
        str: Formatted report string.
    """
    report = []
    report.append("=== 📊 ATS Compatibility Report ===")
    score = analysis["score"]
    score_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
    report.append(f"{score_emoji} Overall Score: {score}/100")

    if analysis["warnings"]:
        report.append("\n⚠️ Warnings:")
        for warning in analysis["warnings"]:
            report.append(f"- {warning}")

    if analysis["recommendations"]:
        report.append("\n💡 Recommendations:")
        for rec in analysis["recommendations"]:
            report.append(f"- {rec}")

    report.append("\n=== 🔍 Detailed Analysis ===")
    details = analysis["details"]

    # Structure
    report.append("\n📑 Structure:")
    structure = details.get("structure", {})
    report.append(f"- Found Sections: {', '.join(structure.get('found_sections', [])) or 'None'}")
    report.append(f"- Missing Sections: {', '.join(structure.get('missing_sections', [])) or 'None'}")
    if "section_order" in structure:
        report.append(f"- Section Order: {' → '.join(structure['section_order'])}")

    # Content
    report.append("\n📝 Content:")
    content = details.get("content", {})
    report.append(f"- Word Count: {content.get('word_count', 0)}")
    report.append(f"- Quantifiable Achievements: {content.get('quantifiables', 0)}")
    if "action_verbs" in content:
        report.append(f"- Action Verbs Used: {content['action_verbs']['count']}")
    if "skills" in content:
        skills = content["skills"][:5] if content["skills"] else []
        report.append(f"- Skills Detected: {', '.join(skills) or 'None'}")

    # Keywords
    report.append("\n🔑 Keywords:")
    keywords = details.get("keywords", {})
    if "similarity_score" in keywords:
        report.append(f"- Job Description Similarity: {keywords['similarity_score']}%")
    if "missing_keywords" in keywords:
        report.append(f"- Missing Keywords: {', '.join(keywords['missing_keywords'][:5]) or 'None'}")

    # Formatting
    report.append("\n🖋️ Formatting:")
    formatting = details.get("formatting", {})
    report.append(f"- Date Formats: {len(formatting.get('date_formats', {}))} types")
    report.append(f"- Bullet Types: {len(formatting.get('bullet_types', {}))} styles")
    if "issues" in formatting:
        report.append(f"- Issues: {', '.join(formatting['issues']) or 'None'}")

    # Status
    status = "ATS-Friendly!" if score >= 80 else "Needs Minor Improvements" if score >= 60 else "Needs Significant Improvements"
    report.append(f"\n🎉 Status: {status}")
    report_str = "\n".join(report)
    logger.info("Generated ATS report")
    return report_str
