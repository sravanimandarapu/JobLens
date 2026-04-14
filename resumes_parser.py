import os
import json
import re
import PyPDF2
import nltk
import spacy
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {str(e)}")

# Load or download spaCy model
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
        raise

# Define dataset paths relative to the script's location
BASE_DIR = os.path.join(os.path.dirname(__file__), "datasets")
SKILLS_FILE = os.path.join(BASE_DIR, "skills.json")
EDUCATION_FILE = os.path.join(BASE_DIR, "education.json")

# Function to load JSON data safely
def load_json(file_path, key):
    if not os.path.exists(file_path):
        logger.error(f"Dataset file not found: {file_path}")
        return {key: []}
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded dataset: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load dataset {file_path}: {str(e)}")
        return {key: []}

# Load skills and education data
skills_set = set(load_json(SKILLS_FILE, "skills").get("skills", []))
education_keywords = set(load_json(EDUCATION_FILE, "education").get("education", []))

# Extract text from a PDF resume
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + " "
        logger.debug(f"Extracted text from PDF: {pdf_path}")
    except Exception as e:
        logger.error(f"PDF extraction failed for {pdf_path}: {str(e)}")
        return {"error": f"PDF extraction failed: {str(e)}"}
    
    return text.strip() if text else {"error": "No text extracted"}

# Enhanced name extraction with better patterns and NER
def extract_name(text, first_n_chars=2000):
    first_portion = text[:first_n_chars]
    doc = nlp(first_portion)
    
    patterns = [
        r"(?i)(?:name\s*(?::|is|:|\-)?\s*)?(?:Dr\.|Mr\.|Ms\.|Mrs\.|Prof\.|Miss)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        r"(?i)(?:name\s*(?::|is|:|\-)?\s*)?(?:Dr\.|Mr\.|Ms\.|Mrs\.|Prof\.|Miss)?\s*([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)",
        r"(?i)name\s*(?::|is|:|\-)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*[\|\-]\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, first_portion, re.MULTILINE)
        if matches:
            name = matches.group(1).strip()
            name = re.sub(r'[^\w\s]', '', name).strip()
            logger.debug(f"Extracted name using regex: {name}")
            return name
    
    candidates = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            if len(ent.text.split()) >= 2 and not ent.text.isupper():
                candidates.append(ent.text)
    
    if candidates:
        name = candidates[0].strip()
        name = re.sub(r'[^\w\s]', '', name).strip()
        logger.debug(f"Extracted name using NER: {name}")
        return name
    
    lines = first_portion.split('\n')[:5]
    for line in lines:
        line = line.strip()
        match = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})", line)
        if match:
            name = match.group(1).strip()
            if not any(keyword in name.lower() for keyword in ["objective", "education", "experience", "skills"]):
                logger.debug(f"Extracted name using fallback: {name}")
                return name
    
    logger.debug("Name not found in resume")
    return "Not Found"

# Enhanced email extraction with better regex and context
def extract_email(text):
    pattern = r"(?i)\b[A-Za-z0-9._%+-]+(?:\+[A-Za-z0-9._%+-]*)?@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    
    email_section = re.search(r"(?:Email|E-mail|Contact)(?::|.)[^\n]*", text, re.IGNORECASE)
    if email_section:
        matches = re.findall(pattern, email_section.group(0))
        if matches:
            email = matches[0].strip()
            logger.debug(f"Extracted email: {email}")
            return email
    
    matches = re.findall(pattern, text)
    if matches:
        email = matches[0].strip()
        logger.debug(f"Extracted email: {email}")
        return email
    
    header = text[:1000]
    matches = re.findall(pattern, header)
    if matches:
        email = matches[0].strip()
        logger.debug(f"Extracted email from header: {email}")
        return email
    
    logger.debug("Email not found in resume")
    return "Not Found"

# Enhanced phone number extraction with better prioritization and cleaning
def extract_contact_number(text):
    patterns = [
        r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        r"\b(?:\+\d{1,3})?\d{10,12}\b",
        r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b",
        r"(?:Phone|Tel|Mobile|Contact)(?::|.)[^\n\d]*(\+?\d[\d\s\-\(\)\.]{8,15}\d)"
    ]
    
    for pattern in patterns[-1:]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            cleaned_number = re.sub(r'[^\d+]', '', matches[0])
            if len(cleaned_number) >= 10:
                logger.debug(f"Extracted contact number: {cleaned_number}")
                return cleaned_number
    
    for pattern in patterns[:-1]:
        matches = re.findall(pattern, text)
        if matches:
            cleaned_number = re.sub(r'[^\d+]', '', matches[0])
            if len(cleaned_number) >= 10:
                header = text[:1000]
                if re.search(pattern, header):
                    logger.debug(f"Extracted contact number from header: {cleaned_number}")
                    return cleaned_number
                logger.debug(f"Extracted contact number: {cleaned_number}")
                return cleaned_number
    
    header = text[:1000]
    match = re.search(r"\b(\+?\d{10,12})\b", header)
    if match:
        number = match.group(1)
        logger.debug(f"Extracted contact number from header fallback: {number}")
        return number
    
    logger.debug("Contact number not found in resume")
    return "Not Found"

# Enhanced skills extraction with better matching
def extract_skills(text):
    text_lower = text.lower()
    found_skills = set()
    
    for skill in skills_set:
        skill_lower = skill.lower()
        if re.search(r'\b' + re.escape(skill_lower) + r'\b', text_lower):
            found_skills.add(skill)
    
    skill_sections = re.findall(r'(?:SKILLS|TECHNICAL SKILLS|EXPERTISE)(?:[^\n]*)([\s\S]*?)(?:EXPERIENCE|EDUCATION|PROJECTS|$)', text, re.IGNORECASE)
    if skill_sections:
        for section in skill_sections:
            items = re.split(r'[,•|\n\-\/]', section)
            for item in items:
                item = item.strip().lower()
                for skill in skills_set:
                    if skill.lower() in item:
                        found_skills.add(skill)
    
    skills_list = list(found_skills) if found_skills else ["Not Found"]
    logger.debug(f"Extracted skills: {skills_list}")
    return skills_list

# Enhanced education extraction
def extract_education(text):
    education_items = []
    
    edu_sections = re.findall(r'(?:EDUCATION|ACADEMIC|QUALIFICATION)(?:[^\n]*)([\s\S]*?)(?:EXPERIENCE|SKILLS|PROJECTS|$)', text, re.IGNORECASE)
    
    if edu_sections:
        for section in edu_sections:
            items = re.split(r'(?:\n\s*\n|\d{4}\s*-\s*\d{4}|\d{4}\s*-\s*Present)', section)
            for item in items:
                if item.strip():
                    degree_match = re.search(r'(Bachelor|Master|B\.Tech|M\.Tech|Intermediate|High School)', item, re.IGNORECASE)
                    college_match = re.search(r'(College|University|Institute|School)', item, re.IGNORECASE)
                    
                    if degree_match or college_match:
                        year_pattern = r'(20\d\d\s*-\s*20\d\d|20\d\d\s*-\s*Present|\d{4}\s*-\s*\d{4})'
                        year_match = re.search(year_pattern, item)
                        year = year_match.group(1) if year_match else ""
                        
                        cleaned_item = item.strip()
                        if year:
                            if year not in cleaned_item:
                                cleaned_item = f"{cleaned_item} {year}"
                        
                        education_items.append(cleaned_item)
    
    if not education_items:
        education_patterns = [
            r"([^.\n]*College[^.\n]*\d{4}[^.\n]*\d{4}[^.\n]*)",
            r"([^.\n]*University[^.\n]*\d{4}[^.\n]*\d{4}[^.\n]*)",
            r"([^.\n]*School[^.\n]*\d{4}[^.\n]*\d{4}[^.\n]*)",
            r"([^.\n]*Bachelor[^.\n]*\d{4}[^.\n]*\d{4}[^.\n]*)",
            r"([^.\n]*B\.Tech[^.\n]*\d{4}[^.\n]*\d{4}[^.\n]*)",
            r"([^.\n]*Intermediate[^.\n]*\d{4}[^.\n]*\d{4}[^.\n]*)",
            r"([^.\n]*High School[^.\n]*\d{4}[^.\n]*\d{4}[^.\n]*)"
        ]
        
        for pattern in education_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                education_items.append(match.strip())
    
    if not education_items:
        year_splits = re.split(r'(20\d\d\s*-\s*20\d\d|20\d\d\s*-\s*Present)', text)
        
        for i in range(1, len(year_splits), 2):
            if i+1 < len(year_splits):
                year = year_splits[i]
                content = year_splits[i+1]
                
                if any(kw in content.lower() for kw in ["college", "university", "school", "bachelor", "b.tech", "intermediate"]):
                    edu_match = re.search(r'([^.\n]{0,100}(college|university|school|bachelor|b\.tech|intermediate)[^.\n]{0,100})', content, re.IGNORECASE)
                    if edu_match:
                        education_items.append(f"{edu_match.group(1).strip()} {year}")
    
    structured_education = []
    for item in education_items:
        parts = re.split(r'(?<=\d{4})', item)
        for part in parts:
            if len(part.strip()) > 10:
                structured_education.append(part.strip())
    
    if structured_education:
        logger.debug(f"Extracted education: {structured_education}")
        return structured_education
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines:
        if any(kw in line.lower() for kw in ["college", "university", "school", "bachelor", "b.tech", "intermediate"]):
            education_items.append(line)
    
    education_list = education_items if education_items else ["Not Found"]
    logger.debug(f"Extracted education: {education_list}")
    return education_list

def extract_experience(text):
    experience_items = []
    
    experience_sections = re.findall(
        r'(?:EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|CAREER)(?:[^\n]*)([\s\S]*?)(?:EDUCATION|SKILLS|PROJECTS|$)', 
        text, 
        re.IGNORECASE
    )
    
    if experience_sections:
        for section in experience_sections:
            job_splits = re.split(r'(?=\d{4}\s*-\s*\d{4}|\d{4}\s*-\s*Present)', section)
            
            for job_entry in job_splits:
                if not job_entry.strip():
                    continue
                
                structured_experience = {
                    "Job Title": "Not Found",
                    "Company": "Not Found",
                    "Duration": "Not Found",
                    "Location": "Not Found",
                    "Key Responsibilities": []
                }
                
                title_patterns = [
                    r'^([A-Za-z\s]+)(?=\s*at|\s*,|\s*@|$)',
                    r'([A-Za-z\s]+)\s*(?:at|@|,)\s*[A-Za-z\s]+',
                ]
                for pattern in title_patterns:
                    title_match = re.search(pattern, job_entry, re.MULTILINE)
                    if title_match:
                        structured_experience["Job Title"] = title_match.group(1).strip()
                        break
                
                company_patterns = [
                    r'(?:at|@|,)\s*([A-Za-z\s&]+)(?=\s*\d{4}|\s*-)',
                    r'[A-Za-z\s]+\s*(?:at|@|,)\s*([A-Za-z\s&]+)'
                ]
                for pattern in company_patterns:
                    company_match = re.search(pattern, job_entry, re.IGNORECASE)
                    if company_match:
                        structured_experience["Company"] = company_match.group(1).strip()
                        break
                
                location_match = re.search(r'([A-Za-z\s]+,\s*[A-Z]{2}|[A-Za-z\s]+)', job_entry)
                if location_match:
                    structured_experience["Location"] = location_match.group(1).strip()
                
                date_match = re.search(r'(\d{4}\s*-\s*\d{4}|\d{4}\s*-\s*Present)', job_entry)
                if date_match:
                    structured_experience["Duration"] = date_match.group(1).strip()
                
                responsibility_patterns = [
                    r'•\s*(.+?)(?=\n•|\n\n|$)',
                    r'[-]\s*(.+?)(?=\n[-]|\n\n|$)',
                    r'^\s*[•-]\s*(.+?)$'
                ]
                
                responsibilities = []
                for pattern in responsibility_patterns:
                    resp_matches = re.findall(pattern, job_entry, re.MULTILINE | re.DOTALL)
                    cleaned_resps = [
                        resp.strip() for resp in resp_matches 
                        if resp.strip() and len(resp.strip()) > 10
                    ]
                    responsibilities.extend(cleaned_resps)
                
                unique_responsibilities = []
                for resp in responsibilities:
                    if resp not in unique_responsibilities:
                        unique_responsibilities.append(resp)
                
                structured_experience["Key Responsibilities"] = unique_responsibilities
                
                if any([
                    structured_experience["Job Title"] != "Not Found", 
                    structured_experience["Company"] != "Not Found", 
                    structured_experience["Key Responsibilities"]
                ]):
                    experience_items.append(structured_experience)
    
    if not experience_items:
        job_patterns = [
            r'([A-Za-z\s]+)\s*(?:at|@)\s*([A-Za-z\s]+)\s*(\d{4}\s*-\s*\d{4}|\d{4}\s*-\s*Present)',
            r'([A-Za-z\s]+)\s*,\s*([A-Za-z\s]+)\s*(\d{4}\s*-\s*\d{4}|\d{4}\s*-\s*Present)'
        ]
        
        for pattern in job_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                experience_items.append({
                    "Job Title": match[0].strip(),
                    "Company": match[1].strip(),
                    "Duration": match[2].strip(),
                    "Location": "Not Found",
                    "Key Responsibilities": []
                })
    
    experience_list = experience_items if experience_items else ["Not Found"]
    logger.debug(f"Extracted experience: {experience_list}")
    return experience_list

def extract_resume_details(file_path):
    text = extract_text_from_pdf(file_path)
    if isinstance(text, dict):
        logger.error(f"Failed to extract resume text: {text.get('error')}")
        return text
    
    education_list = extract_education(text)
    
    formatted_education = []
    for edu_item in education_list:
        formatted_edu = edu_item.strip()
        formatted_edu = re.sub(r'\s+', ' ', formatted_edu)
        formatted_education.append(formatted_edu)
    
    experience_list = extract_experience(text)
    
    resume_data = {
        "Name": extract_name(text),
        "Email": extract_email(text),
        "Contact Number": extract_contact_number(text),
        "Skills": extract_skills(text),
        "Education": formatted_education,
        "Experience": experience_list,
        "Raw Text": text[:200] + "..."
    }
    
    logger.info(f"Extracted resume details: {list(resume_data.keys())}")
    return resume_data
